from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import MDS

from llm_geometry.io_utils import load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part 4 Gemma drift cartography over generations.")
    p.add_argument("--models-config", default="configs/models_gemma_part3.yaml")
    p.add_argument("--analysis-config", default="configs/analysis_gemma_part4.yaml")
    p.add_argument("--reports-dir", default="outputs/reports_gemma_part3_360")
    p.add_argument("--out-dir", default="outputs/reports_gemma_part4")
    p.add_argument("--out-md", default="docs/GEMMA_PART4_DRIFT_CARTOGRAPHY.md")
    return p.parse_args()


def pair_lookup(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return df[((df["model_a"] == a) & (df["model_b"] == b)) | ((df["model_a"] == b) & (df["model_b"] == a))]


def safe_mean(df: pd.DataFrame, col: str) -> float:
    if len(df) == 0 or col not in df.columns:
        return float("nan")
    return float(df[col].mean())


def minmax(s: pd.Series) -> pd.Series:
    x = s.astype(float)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(x), dtype=float), index=s.index)
    return (x - lo) / (hi - lo)


def lineage_edges(path: list[str], edge_type: str) -> list[tuple[str, str, str]]:
    return [(path[i], path[i + 1], edge_type) for i in range(len(path) - 1)]


def main() -> None:
    args = parse_args()
    reports = Path(args.reports_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_yaml(args.models_config)
    a_cfg = load_yaml(args.analysis_config)["part4"]

    meta = pd.DataFrame(model_cfg["models"])
    meta = meta[["name", "family", "generation", "tune_type", "params_b", "repo_id"]].rename(columns={"name": "model"})
    meta_idx = meta.set_index("model")

    pair = pd.read_csv(reports / "pairwise_similarity.csv")
    domain = pd.read_csv(reports / "domain_similarity.csv")
    weight = pd.read_csv(reports / "weight_pairwise_drift.csv")

    agg = pair.groupby(["model_a", "model_b"], as_index=False).agg(
        mean_cka=("cka", "mean"),
        mean_rsa=("rsa_spearman", "mean"),
        mean_knn=("knn_overlap", "mean"),
        mean_proc=("procrustes_residual", "mean"),
    )
    agg["rep_drift"] = 1.0 - agg["mean_cka"]

    dom = domain.groupby(["model_a", "model_b"], as_index=False).agg(domain_cka_std=("cka", "std"))
    agg = agg.merge(dom, on=["model_a", "model_b"], how="left")
    agg = agg.merge(weight[["model_a", "model_b", "weight_cosine", "weight_l2"]], on=["model_a", "model_b"], how="left")
    agg["weight_drift"] = 1.0 - agg["weight_cosine"]
    agg["rsa_drift"] = 1.0 - agg["mean_rsa"]
    agg["knn_drift"] = 1.0 - agg["mean_knn"]

    gemma_models = a_cfg["gemma_models"]
    lfm_models = a_cfg["lfm_models"]
    all_focus = set(gemma_models + lfm_models)
    agg = agg[agg["model_a"].isin(all_focus) & agg["model_b"].isin(all_focus)].copy()

    gemma_pairs = agg[agg["model_a"].isin(gemma_models) & agg["model_b"].isin(gemma_models)].copy()
    if len(gemma_pairs) == 0:
        raise RuntimeError("No Gemma-only pairs found in reports.")

    gemma_pairs["n_rep"] = minmax(gemma_pairs["rep_drift"])
    gemma_pairs["n_rsa"] = minmax(gemma_pairs["rsa_drift"])
    gemma_pairs["n_knn"] = minmax(gemma_pairs["knn_drift"])
    gemma_pairs["n_proc"] = minmax(gemma_pairs["mean_proc"])
    gemma_pairs["n_wdrift"] = minmax(gemma_pairs["weight_drift"].fillna(gemma_pairs["weight_drift"].mean()))
    gemma_pairs["drift_index"] = (
        0.40 * gemma_pairs["n_rep"]
        + 0.20 * gemma_pairs["n_rsa"]
        + 0.15 * gemma_pairs["n_knn"]
        + 0.15 * gemma_pairs["n_proc"]
        + 0.10 * gemma_pairs["n_wdrift"]
    )

    pair_index = gemma_pairs.set_index(["model_a", "model_b"])
    pair_index_rev = gemma_pairs.set_index(["model_b", "model_a"])

    def edge_metrics(a: str, b: str) -> pd.Series:
        if (a, b) in pair_index.index:
            row = pair_index.loc[(a, b)]
        elif (a, b) in pair_index_rev.index:
            row = pair_index_rev.loc[(a, b)]
        else:
            raise KeyError(f"Pair not found: {a}, {b}")
        return row

    # Construct lineage and tuning edges
    edges: list[tuple[str, str, str]] = []
    edges += lineage_edges(a_cfg["it_path"], "it_lineage")
    edges += lineage_edges(a_cfg["pt_path"], "pt_lineage")
    edges += [(a, b, "tune_shift") for a, b in a_cfg["tune_pairs"]]

    edge_rows = []
    for a, b, etype in edges:
        r = edge_metrics(a, b)
        ga = float(meta_idx.loc[a, "generation"])
        gb = float(meta_idx.loc[b, "generation"])
        generation_step = abs(gb - ga)
        drift_velocity = float(r["rep_drift"] / generation_step) if generation_step > 0 else np.nan
        edge_rows.append(
            {
                "edge_type": etype,
                "model_from": a,
                "model_to": b,
                "gen_from": ga,
                "gen_to": gb,
                "generation_step": generation_step,
                "rep_drift": float(r["rep_drift"]),
                "rsa_drift": float(r["rsa_drift"]),
                "knn_drift": float(r["knn_drift"]),
                "procrustes": float(r["mean_proc"]),
                "domain_cka_std": float(r["domain_cka_std"]),
                "weight_drift": float(r["weight_drift"]),
                "weight_l2": float(r["weight_l2"]),
                "drift_index": float(r["drift_index"]),
                "drift_velocity": drift_velocity,
            }
        )
    edge_df = pd.DataFrame(edge_rows)

    # Instruction tuning shift over generations
    tune_rows = []
    for pt, it in a_cfg["tune_pairs"]:
        r = edge_metrics(pt, it)
        gen = float(meta_idx.loc[pt, "generation"])
        tune_rows.append(
            {
                "generation": gen,
                "pt_model": pt,
                "it_model": it,
                "rep_drift": float(r["rep_drift"]),
                "rsa_drift": float(r["rsa_drift"]),
                "knn_drift": float(r["knn_drift"]),
                "weight_drift": float(r["weight_drift"]),
                "drift_index": float(r["drift_index"]),
            }
        )
    tune_df = pd.DataFrame(tune_rows).sort_values("generation").reset_index(drop=True)
    tune_df["rep_drift_delta_vs_prev"] = tune_df["rep_drift"].diff()
    tune_df["drift_index_delta_vs_prev"] = tune_df["drift_index"].diff()

    # Gemma model-level drift profile and LFM anchoring
    model_rows = []
    for m in gemma_models:
        g_pairs = gemma_pairs[(gemma_pairs["model_a"] == m) | (gemma_pairs["model_b"] == m)]
        l_pairs = agg[
            ((agg["model_a"] == m) & (agg["model_b"].isin(lfm_models)))
            | ((agg["model_b"] == m) & (agg["model_a"].isin(lfm_models)))
        ]
        model_rows.append(
            {
                "model": m,
                "generation": float(meta_idx.loc[m, "generation"]),
                "tune_type": str(meta_idx.loc[m, "tune_type"]),
                "params_b": float(meta_idx.loc[m, "params_b"]),
                "mean_gemma_rep_drift": float(g_pairs["rep_drift"].mean()),
                "mean_gemma_drift_index": float(g_pairs["drift_index"].mean()),
                "mean_lfm_rep_drift": float(l_pairs["rep_drift"].mean()),
                "mean_lfm_weight_drift": float(l_pairs["weight_drift"].mean()),
            }
        )
    model_df = pd.DataFrame(model_rows)

    # Distance matrix + 2D embedding for Gemma trajectory visualization
    dist = pd.DataFrame(np.zeros((len(gemma_models), len(gemma_models))), index=gemma_models, columns=gemma_models)
    for i, a in enumerate(gemma_models):
        for j, b in enumerate(gemma_models):
            if i == j:
                dist.loc[a, b] = 0.0
                continue
            r = edge_metrics(a, b)
            dist.loc[a, b] = float(r["rep_drift"])

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=4, normalized_stress="auto")
    coords = mds.fit_transform(dist.values)
    emb = pd.DataFrame({"model": gemma_models, "x": coords[:, 0], "y": coords[:, 1]})
    emb = emb.merge(model_df[["model", "generation", "tune_type", "params_b"]], on="model", how="left")
    centroid_x = float(emb["x"].mean())
    centroid_y = float(emb["y"].mean())
    emb["radius_from_centroid"] = np.sqrt((emb["x"] - centroid_x) ** 2 + (emb["y"] - centroid_y) ** 2)
    emb["angle"] = np.degrees(np.arctan2(emb["y"] - centroid_y, emb["x"] - centroid_x))

    # Path accumulation summaries
    def summarize_path(path: list[str], label: str) -> dict[str, float]:
        d = 0.0
        idx = 0.0
        vel = []
        for i in range(len(path) - 1):
            r = edge_metrics(path[i], path[i + 1])
            d += float(r["rep_drift"])
            idx += float(r["drift_index"])
            g0 = float(meta_idx.loc[path[i], "generation"])
            g1 = float(meta_idx.loc[path[i + 1], "generation"])
            if g1 > g0:
                vel.append(float(r["rep_drift"] / (g1 - g0)))
        return {
            "path_label": label,
            "steps": len(path) - 1,
            "cumulative_rep_drift": d,
            "cumulative_drift_index": idx,
            "mean_velocity": float(np.mean(vel)) if vel else np.nan,
            "max_velocity": float(np.max(vel)) if vel else np.nan,
        }

    path_df = pd.DataFrame(
        [
            summarize_path(a_cfg["it_path"], "it_path"),
            summarize_path(a_cfg["pt_path"], "pt_path"),
        ]
    )

    # Save artifacts
    gemma_pairs_out = gemma_pairs[
        [
            "model_a",
            "model_b",
            "rep_drift",
            "rsa_drift",
            "knn_drift",
            "mean_proc",
            "domain_cka_std",
            "weight_drift",
            "weight_l2",
            "drift_index",
        ]
    ].copy()
    gemma_pairs_out = gemma_pairs_out.rename(columns={"mean_proc": "procrustes"})
    gemma_pairs_out.to_csv(out_dir / "part4_pair_drift_matrix.csv", index=False)
    edge_df.to_csv(out_dir / "part4_lineage_edges.csv", index=False)
    tune_df.to_csv(out_dir / "part4_instruction_shift.csv", index=False)
    model_df.to_csv(out_dir / "part4_model_drift_profile.csv", index=False)
    emb.to_csv(out_dir / "part4_model_embedding.csv", index=False)
    path_df.to_csv(out_dir / "part4_path_summary.csv", index=False)

    # Markdown report
    top_edges = edge_df.sort_values("drift_index", ascending=False).head(12)
    stable_edges = edge_df.sort_values("drift_index", ascending=True).head(12)
    lines = []
    lines.append("# Gemma Part 4: Drift Cartography")
    lines.append("")
    lines.append("## Framework")
    lines.append("Part 4 maps temporal drift using a composite drift index over representation, neighborhood, structural, and weight-space components.")
    lines.append("")
    lines.append("Composite drift index weights:")
    lines.append("- rep_drift (1-CKA): 0.40")
    lines.append("- rsa_drift (1-RSA): 0.20")
    lines.append("- knn_drift (1-KNN overlap): 0.15")
    lines.append("- procrustes residual: 0.15")
    lines.append("- weight_drift (1-cosine): 0.10")
    lines.append("")
    lines.append("## Path Summary")
    lines.append(path_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Lineage + Tuning Edges")
    lines.append(edge_df.sort_values(["edge_type", "gen_from"]).to_markdown(index=False))
    lines.append("")
    lines.append("## Instruction Shift Over Time")
    lines.append(tune_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Model Drift Profile")
    lines.append(model_df.sort_values("generation").to_markdown(index=False))
    lines.append("")
    lines.append("## Highest Drift Edges")
    lines.append(top_edges.to_markdown(index=False))
    lines.append("")
    lines.append("## Lowest Drift Edges")
    lines.append(stable_edges.to_markdown(index=False))
    lines.append("")
    lines.append("## Conclusions")
    lines.append("- Gemma drift is dominated by generation transitions and within-generation PT->IT tuning shifts at 3B-era checkpoints.")
    lines.append("- Instruction tuning drift is not monotonic across generations; it should be tracked as a generation-specific operator, not a constant offset.")
    lines.append("- LFM anchors remain useful as an external baseline for measuring how far Gemma checkpoints move from compact-model geometry priors.")

    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
