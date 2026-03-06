from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize Gemma 3 4B vs 27B comparison outputs.")
    p.add_argument("--reports-dir", default="outputs/reports_gemma_4b_27b_24")
    p.add_argument("--prompt-csv", default="data/prompts_gemma_4b_27b_24.csv")
    p.add_argument("--out-md", default="docs/GEMMA_4B_27B_COMPARISON.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reports = Path(args.reports_dir)
    out_md = Path(args.out_md)

    pair_path = reports / "pairwise_similarity.csv"
    domain_path = reports / "domain_similarity.csv"
    weight_path = reports / "weight_pairwise_drift.csv"
    stats_path = reports / "weight_fingerprint_stats.csv"
    layer_geom_path = reports / "weight_layer_geometry.csv"
    layer_geom_summary_path = reports / "weight_layer_geometry_summary.csv"

    lines = []
    lines.append("# Gemma 3 4B vs 27B Comparison")
    lines.append("")
    p = Path(args.prompt_csv)
    n_prompts = None
    n_domains = None
    if p.exists():
        pdf = pd.read_csv(p)
        n_prompts = len(pdf)
        n_domains = int(pdf["domain"].nunique()) if "domain" in pdf.columns else None

    lines.append("## Setup")
    lines.append("- Models: `google/gemma-3-4b-it` vs `google/gemma-3-27b-it`")
    if n_prompts is not None and n_domains is not None:
        lines.append(f"- Prompt set: {n_prompts} prompts across {n_domains} domains")
    elif n_prompts is not None:
        lines.append(f"- Prompt set: {n_prompts} prompts")
    lines.append("- Layers sampled: 0, 6, 12")
    lines.append("")

    if pair_path.exists():
        pair = pd.read_csv(pair_path)
        if len(pair):
            overall = pair.agg(
                mean_cka=("cka", "mean"),
                mean_rsa=("rsa_spearman", "mean"),
                mean_knn=("knn_overlap", "mean"),
                mean_procrustes=("procrustes_residual", "mean"),
            )
            overall_df = pd.DataFrame([overall.to_dict()])
            overall_df["rep_drift_1_minus_cka"] = 1.0 - overall_df["mean_cka"]
            lines.append("## Overall Representation Similarity")
            lines.append(overall_df.to_markdown(index=False))
            lines.append("")

            layer_df = pair.groupby("layer", as_index=False).agg(
                mean_cka=("cka", "mean"),
                mean_rsa=("rsa_spearman", "mean"),
                mean_knn=("knn_overlap", "mean"),
                mean_procrustes=("procrustes_residual", "mean"),
            )
            layer_df["rep_drift_1_minus_cka"] = 1.0 - layer_df["mean_cka"]
            lines.append("## Layerwise Similarity")
            lines.append(layer_df.sort_values("layer").to_markdown(index=False))
            lines.append("")
        else:
            lines.append("## Representation Similarity")
            lines.append("No pairwise similarity rows were generated.")
            lines.append("")
    else:
        lines.append("## Representation Similarity")
        lines.append("Geometry comparison artifacts were not generated (likely due extraction constraints for 27B in this environment).")
        lines.append("")

    if domain_path.exists():
        ddf = pd.read_csv(domain_path)
        if len(ddf):
            dsum = ddf.groupby("domain", as_index=False).agg(
                mean_cka=("cka", "mean"),
                mean_rsa=("rsa_spearman", "mean"),
                mean_knn=("knn_overlap", "mean"),
                mean_procrustes=("procrustes_residual", "mean"),
            )
            dsum["rep_drift_1_minus_cka"] = 1.0 - dsum["mean_cka"]
            lines.append("## Domain Breakdown")
            lines.append(dsum.sort_values("rep_drift_1_minus_cka", ascending=False).to_markdown(index=False))
            lines.append("")

    if weight_path.exists():
        wdf = pd.read_csv(weight_path)
        if len(wdf):
            wdf = wdf.copy()
            wdf["weight_drift_1_minus_cosine"] = 1.0 - wdf["weight_cosine"]
            lines.append("## Weight Fingerprint Drift")
            lines.append(wdf.to_markdown(index=False))
            lines.append("")
    if stats_path.exists():
        sdf = pd.read_csv(stats_path)
        if len(sdf):
            lines.append("## Fingerprint Coverage")
            lines.append(sdf[["model", "tensors_used", "global_n", "global_rms", "global_abs_mean"]].to_markdown(index=False))
            lines.append("")

    if layer_geom_summary_path.exists():
        gsum = pd.read_csv(layer_geom_summary_path)
        if len(gsum):
            lines.append("## Layerwise Weight-Geometry Summary")
            lines.append(gsum.to_markdown(index=False))
            lines.append("")

    if layer_geom_path.exists():
        gdf = pd.read_csv(layer_geom_path)
        if len(gdf):
            lines.append("## Layerwise Weight-Geometry Mapping")
            lines.append(gdf.to_markdown(index=False))
            lines.append("")
    else:
        lines.append("## Weight Fingerprint Drift")
        lines.append("No weight drift artifact found.")
        lines.append("")

    lines.append("## Notes")
    lines.append("- If representation artifacts are missing while weight drift is present, the 27B checkpoint likely exceeded runtime/memory for hidden-state extraction.")
    lines.append("- Weight drift still gives a robust first-order scale comparison between 4B and 27B checkpoints.")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
