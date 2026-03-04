from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from llm_geometry.io_utils import load_yaml


SIZE_ORDER = ["1B", "3-5B", "8B", "other"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build detailed drift/variance report for Part 2 model set.")
    p.add_argument("--models-config", default="configs/models_part2.yaml")
    p.add_argument("--reports-dir", default="outputs/reports_part2_360")
    p.add_argument("--out-md", default="docs/PART2_DRIFT_REPORT.md")
    return p.parse_args()


def bucket_from_params(params_b: float) -> str:
    if params_b <= 1.5:
        return "1B"
    if 2.5 <= params_b <= 5.5:
        return "3-5B"
    if 6.5 <= params_b <= 9.5:
        return "8B"
    return "other"


def main() -> None:
    args = parse_args()
    mcfg = load_yaml(args.models_config)
    reports_dir = Path(args.reports_dir)

    pair = pd.read_csv(reports_dir / "pairwise_similarity.csv")
    domain = pd.read_csv(reports_dir / "domain_similarity.csv")
    ci = pd.read_csv(reports_dir / "pairwise_bootstrap_ci.csv")
    summary = pd.read_csv(reports_dir / "geometry_summary.csv")
    perm = pd.read_csv(reports_dir / "permutation_tests.csv")

    model_meta = {}
    for m in mcfg["models"]:
        if m.get("skip_extraction", False):
            continue
        model_meta[m["name"]] = {
            "params_b": float(m.get("params_b", 0.0)),
            "size_bucket": m.get("size_bucket") or bucket_from_params(float(m.get("params_b", 0.0))),
            "family": m.get("family", "unknown"),
        }

    def map_bucket(model_name: str) -> str:
        return model_meta.get(model_name, {}).get("size_bucket", "other")

    pair["bucket_a"] = pair["model_a"].map(map_bucket)
    pair["bucket_b"] = pair["model_b"].map(map_bucket)
    pair["bucket_relation"] = np.where(pair["bucket_a"] == pair["bucket_b"], "within", "cross")
    pair["pair_label"] = pair["model_a"] + " vs " + pair["model_b"]

    within_cross = (
        pair.groupby(["bucket_relation", "layer"], as_index=False)
        .agg(
            mean_cka=("cka", "mean"),
            std_cka=("cka", "std"),
            mean_rsa=("rsa_spearman", "mean"),
            mean_knn=("knn_overlap", "mean"),
            mean_proc=("procrustes_residual", "mean"),
            n=("cka", "size"),
        )
        .sort_values(["bucket_relation", "layer"])
    )

    bucket_matrix = (
        pair.groupby(["bucket_a", "bucket_b", "layer"], as_index=False)
        .agg(mean_cka=("cka", "mean"), mean_rsa=("rsa_spearman", "mean"), n=("cka", "size"))
        .sort_values(["bucket_a", "bucket_b", "layer"])
    )

    depth = pair.copy()
    depth["layer_id"] = depth["layer"].str.replace("layer_", "", regex=False).astype(int)
    shallow = depth[depth["layer_id"] == depth["layer_id"].min()][["pair_label", "cka"]].rename(columns={"cka": "cka_shallow"})
    deep = depth[depth["layer_id"] == depth["layer_id"].max()][["pair_label", "cka"]].rename(columns={"cka": "cka_deep"})
    depth_drift = shallow.merge(deep, on="pair_label", how="inner")
    depth_drift["depth_drift"] = depth_drift["cka_shallow"] - depth_drift["cka_deep"]

    domain_var = (
        domain.groupby(["model_a", "model_b", "layer"], as_index=False)
        .agg(domain_cka_mean=("cka", "mean"), domain_cka_std=("cka", "std"))
    )
    domain_var["pair_label"] = domain_var["model_a"] + " vs " + domain_var["model_b"]

    perm_sig_rate = float((perm["p_value"] < 0.05).mean()) if len(perm) else float("nan")

    ci_cka = ci[ci["metric"] == "cka"].copy()
    ci_cka["ci_width"] = ci_cka["ci_high"] - ci_cka["ci_low"]

    pair_rank = pair.groupby("pair_label", as_index=False).agg(
        mean_cka=("cka", "mean"),
        mean_rsa=("rsa_spearman", "mean"),
        mean_knn=("knn_overlap", "mean"),
        mean_proc=("procrustes_residual", "mean"),
    ).sort_values("mean_cka", ascending=False)

    model_health = summary.groupby("model", as_index=False).agg(
        mean_anisotropy=("anisotropy", "mean"),
        mean_participation_ratio=("participation_ratio", "mean"),
    ).sort_values("mean_anisotropy")

    top_drift = depth_drift.sort_values("depth_drift", ascending=False).head(12)
    top_domain_var = domain_var.sort_values("domain_cka_std", ascending=False).head(12)

    lines = []
    lines.append("# Part 2 Drift and Variance Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("This report compares requested Part 2 models across parameter-scale groups (1B, 3-5B, 8B, other) using CKA/RSA/topology metrics over sampled layers.")
    lines.append("")
    lines.append("## Model Inventory")
    lines.append(pd.DataFrame([
        {"model": k, **v} for k, v in model_meta.items()
    ]).sort_values(["size_bucket", "params_b", "model"]).to_markdown(index=False))
    lines.append("")
    lines.append("## Pairwise Ranking (Mean Across Layers)")
    lines.append(pair_rank.to_markdown(index=False))
    lines.append("")
    lines.append("## Within-vs-Cross Bucket Alignment")
    lines.append(within_cross.to_markdown(index=False))
    lines.append("")
    lines.append("## Bucket-to-Bucket Matrix (Layer-Conditioned)")
    lines.append(bucket_matrix.to_markdown(index=False))
    lines.append("")
    lines.append("## Depth Drift (Shallow-to-Deep CKA Drop)")
    lines.append("Higher value means stronger drift from early to late layers.")
    lines.append(top_drift.to_markdown(index=False))
    lines.append("")
    lines.append("## Domain Variance Hotspots")
    lines.append("Higher domain_cka_std means stronger domain sensitivity (potential context-conditioned drift).")
    lines.append(top_domain_var.to_markdown(index=False))
    lines.append("")
    lines.append("## Geometry Health by Model")
    lines.append(model_health.to_markdown(index=False))
    lines.append("")
    lines.append("## Statistical Stability")
    lines.append(f"- Permutation significance rate (p<0.05): {perm_sig_rate:.4f}")
    if len(ci_cka):
        lines.append(f"- Mean CKA CI width: {ci_cka['ci_width'].mean():.4f}")
        lines.append(f"- 90th percentile CKA CI width: {ci_cka['ci_width'].quantile(0.9):.4f}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- Compare within-bucket vs cross-bucket CKA by layer to separate size-driven variance from family-driven variance.")
    lines.append("- Large depth drift with high domain variance indicates unstable geometric transfer under changing task manifolds.")
    lines.append("- High anisotropy with low participation ratio suggests representation concentration that can increase brittle behavior.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- reports: `{reports_dir}`")
    lines.append(f"- main visualization: `{reports_dir / 'geometry_atlas_part2_focus.html'}`")

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
