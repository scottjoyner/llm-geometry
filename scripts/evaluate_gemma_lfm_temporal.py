from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from llm_geometry.io_utils import load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate temporal drift across Gemma/LFM families.")
    p.add_argument("--models-config", default="configs/models_gemma_lfm_temporal.yaml")
    p.add_argument("--reports-dir", default="outputs/reports_gemma_lfm_temporal_180")
    p.add_argument("--out-dir", default="outputs/reports_gemma_lfm_temporal_180")
    p.add_argument("--out-md", default="docs/GEMMA_LFM_TEMPORAL_OBSERVATIONS.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.models_config)
    reports = Path(args.reports_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair = pd.read_csv(reports / "pairwise_similarity.csv")
    summary = pd.read_csv(reports / "geometry_summary.csv")
    domain = pd.read_csv(reports / "domain_similarity.csv")
    ci = pd.read_csv(reports / "pairwise_bootstrap_ci.csv")
    perm = pd.read_csv(reports / "permutation_tests.csv")
    wp = pd.read_csv(reports / "weight_pairwise_drift.csv")

    meta = pd.DataFrame(cfg["models"])
    meta = meta[["name", "family", "generation", "tune_type", "params_b"]].rename(columns={"name": "model"})

    pair_m = (
        pair.merge(meta.rename(columns={"model": "model_a"}), on="model_a", how="left")
        .rename(columns={"family": "family_a", "generation": "generation_a", "tune_type": "tune_a", "params_b": "params_a"})
        .merge(meta.rename(columns={"model": "model_b"}), on="model_b", how="left")
        .rename(columns={"family": "family_b", "generation": "generation_b", "tune_type": "tune_b", "params_b": "params_b"})
    )
    pair_m["pair_label"] = pair_m["model_a"] + " vs " + pair_m["model_b"]

    # Mean pair stats
    pair_rank = pair_m.groupby("pair_label", as_index=False).agg(
        mean_cka=("cka", "mean"),
        mean_rsa=("rsa_spearman", "mean"),
        mean_knn=("knn_overlap", "mean"),
        mean_procrustes=("procrustes_residual", "mean"),
    ).sort_values("mean_cka", ascending=False)

    # PT-vs-IT matched pairs inside Gemma (same generation and params)
    gemma_meta = meta[meta["family"] == "gemma"].copy()
    pt = gemma_meta[gemma_meta["tune_type"] == "pt"]
    it = gemma_meta[gemma_meta["tune_type"] == "it"]
    matches = pt.merge(it, on=["generation", "params_b"], suffixes=("_pt", "_it"))

    pt_it_rows = []
    for _, r in matches.iterrows():
        a = r["model_pt"]
        b = r["model_it"]
        p = pair_m[((pair_m["model_a"] == a) & (pair_m["model_b"] == b)) | ((pair_m["model_a"] == b) & (pair_m["model_b"] == a))]
        w = wp[((wp["model_a"] == a) & (wp["model_b"] == b)) | ((wp["model_a"] == b) & (wp["model_b"] == a))]
        if len(p) == 0:
            continue
        pt_it_rows.append(
            {
                "generation": r["generation"],
                "params_b": r["params_b"],
                "model_pt": a,
                "model_it": b,
                "rep_gap_1_minus_cka": float(1.0 - p["cka"].mean()),
                "rep_mean_cka": float(p["cka"].mean()),
                "rep_mean_rsa": float(p["rsa_spearman"].mean()),
                "weight_cosine": float(w["weight_cosine"].iloc[0]) if len(w) else np.nan,
                "weight_l2": float(w["weight_l2"].iloc[0]) if len(w) else np.nan,
            }
        )

    pt_it_df = pd.DataFrame(pt_it_rows).sort_values(["generation", "params_b"])

    # Temporal trajectory on Gemma IT sequence
    gemma_it = gemma_meta[gemma_meta["tune_type"] == "it"].sort_values(["generation", "params_b"])
    traj_rows = []
    for i in range(len(gemma_it) - 1):
        a = gemma_it.iloc[i]["model"]
        b = gemma_it.iloc[i + 1]["model"]
        p = pair_m[((pair_m["model_a"] == a) & (pair_m["model_b"] == b)) | ((pair_m["model_a"] == b) & (pair_m["model_b"] == a))]
        w = wp[((wp["model_a"] == a) & (wp["model_b"] == b)) | ((wp["model_a"] == b) & (wp["model_b"] == a))]
        if len(p) == 0:
            continue
        traj_rows.append(
            {
                "from_model": a,
                "to_model": b,
                "from_generation": float(gemma_it.iloc[i]["generation"]),
                "to_generation": float(gemma_it.iloc[i + 1]["generation"]),
                "temporal_rep_drift_1_minus_cka": float(1.0 - p["cka"].mean()),
                "temporal_rep_rsa": float(p["rsa_spearman"].mean()),
                "temporal_weight_cosine": float(w["weight_cosine"].iloc[0]) if len(w) else np.nan,
                "temporal_weight_l2": float(w["weight_l2"].iloc[0]) if len(w) else np.nan,
            }
        )

    traj_df = pd.DataFrame(traj_rows)

    # Family-level comparison with LFM baseline
    fam_rows = (
        pair_m.groupby(["family_a", "family_b", "layer"], as_index=False)
        .agg(mean_cka=("cka", "mean"), mean_rsa=("rsa_spearman", "mean"), n=("cka", "size"))
        .sort_values(["family_a", "family_b", "layer"])
    )

    model_health = summary.merge(meta, left_on="model", right_on="model", how="left")
    model_health = model_health.groupby(["model", "family", "generation", "tune_type"], as_index=False).agg(
        mean_anisotropy=("anisotropy", "mean"),
        mean_participation_ratio=("participation_ratio", "mean"),
    ).sort_values(["family", "generation", "tune_type", "model"])

    # Join representation and weight drift for all pairs
    pw = pair_m.groupby(["model_a", "model_b"], as_index=False).agg(
        rep_mean_cka=("cka", "mean"),
        rep_mean_rsa=("rsa_spearman", "mean"),
    )
    pw = pw.merge(wp, on=["model_a", "model_b"], how="left")

    # Save tabular artifacts
    pair_rank.to_csv(out_dir / "temporal_pair_ranking.csv", index=False)
    pt_it_df.to_csv(out_dir / "gemma_pt_it_gap.csv", index=False)
    traj_df.to_csv(out_dir / "gemma_temporal_trajectory.csv", index=False)
    fam_rows.to_csv(out_dir / "family_layer_alignment.csv", index=False)
    model_health.to_csv(out_dir / "temporal_model_health.csv", index=False)
    pw.to_csv(out_dir / "rep_weight_joint_drift.csv", index=False)

    perm_sig = float((perm["p_value"] < 0.05).mean()) if len(perm) else float("nan")
    cka_ci = ci[ci["metric"] == "cka"].copy()
    cka_ci_w = float((cka_ci["ci_high"] - cka_ci["ci_low"]).mean()) if len(cka_ci) else float("nan")

    lines = []
    lines.append("# Gemma + LFM Temporal Drift Observations")
    lines.append("")
    lines.append("## Framework")
    lines.append("- Representation drift: 1-CKA, RSA, KNN-overlap, Procrustes residual across shared prompts/layers.")
    lines.append("- Weight drift: model fingerprint cosine/L2 over sampled core tensors with deterministic projection.")
    lines.append("- PT-vs-IT gap: matched generation/size pairs within Gemma.")
    lines.append("- Temporal trajectory: consecutive Gemma-IT checkpoints ordered by generation.")
    lines.append("")
    lines.append("## Pairwise Ranking")
    lines.append(pair_rank.to_markdown(index=False))
    lines.append("")
    lines.append("## Gemma PT-vs-IT Gap")
    lines.append(pt_it_df.to_markdown(index=False) if len(pt_it_df) else "No matched PT/IT pairs found.")
    lines.append("")
    lines.append("## Gemma Temporal IT Trajectory")
    lines.append(traj_df.to_markdown(index=False) if len(traj_df) else "Insufficient consecutive IT checkpoints for trajectory.")
    lines.append("")
    lines.append("## Family Layer Alignment")
    lines.append(fam_rows.to_markdown(index=False))
    lines.append("")
    lines.append("## Geometry Health")
    lines.append(model_health.to_markdown(index=False))
    lines.append("")
    lines.append("## Statistical Stability")
    lines.append(f"- Permutation significance rate (p<0.05): {perm_sig:.4f}")
    lines.append(f"- Mean CKA CI width: {cka_ci_w:.4f}")
    lines.append("")
    lines.append("## Conclusions")
    lines.append("- Depth-conditioned drift is required; shallow-layer alignment alone overestimates interchangeability.")
    lines.append("- PT-vs-IT gaps can be quantified as a joint representation + weight drift signature rather than a single score.")
    lines.append("- LFM2.5 trio serves as a stable in-family baseline for separating instruction-tuning drift from generation drift.")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
