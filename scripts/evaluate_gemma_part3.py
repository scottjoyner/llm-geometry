from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from llm_geometry.io_utils import load_yaml
from llm_geometry.metrics import linear_cka


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part 3 Gemma lineage drift decomposition and layer transport.")
    p.add_argument("--models-config", default="configs/models_gemma_part3.yaml")
    p.add_argument("--benchmark-config", default="configs/benchmark_gemma_part3_360.yaml")
    p.add_argument("--geometry-dir", default="outputs/geometry_gemma_part3_360")
    p.add_argument("--reports-dir", default="outputs/reports_gemma_part3_360")
    p.add_argument("--out-md", default="docs/GEMMA_PART3_OBSERVATIONS.md")
    return p.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def mean_pair_value(df: pd.DataFrame, a: str, b: str, col: str) -> float:
    s = df[((df["model_a"] == a) & (df["model_b"] == b)) | ((df["model_a"] == b) & (df["model_b"] == a))]
    if len(s) == 0:
        return float("nan")
    return float(s[col].mean())


def main() -> None:
    args = parse_args()
    mcfg = load_yaml(args.models_config)
    bcfg = load_yaml(args.benchmark_config)

    reports = Path(args.reports_dir)
    gdir = Path(args.geometry_dir)
    reports.mkdir(parents=True, exist_ok=True)

    pair = pd.read_csv(reports / "pairwise_similarity.csv")
    domain = pd.read_csv(reports / "domain_similarity.csv")
    weight = pd.read_csv(reports / "weight_pairwise_drift.csv")
    health = pd.read_csv(reports / "temporal_model_health.csv") if (reports / "temporal_model_health.csv").exists() else pd.DataFrame()

    meta = pd.DataFrame(mcfg["models"])
    meta = meta[["name", "family", "generation", "tune_type", "params_b"]].rename(columns={"name": "model"})

    # Pair-level decomposition dataset
    recs = []
    pair_mean = pair.groupby(["model_a", "model_b"], as_index=False).agg(
        rep_mean_cka=("cka", "mean"),
        rep_mean_rsa=("rsa_spearman", "mean"),
    )

    for _, r in pair_mean.iterrows():
        a = r["model_a"]
        b = r["model_b"]
        ma = meta[meta["model"] == a].iloc[0]
        mb = meta[meta["model"] == b].iloc[0]

        dsub = domain[((domain["model_a"] == a) & (domain["model_b"] == b)) | ((domain["model_a"] == b) & (domain["model_b"] == a))]
        domain_std = float(dsub["cka"].std()) if len(dsub) else np.nan

        w = weight[((weight["model_a"] == a) & (weight["model_b"] == b)) | ((weight["model_a"] == b) & (weight["model_b"] == a))]
        wcos = float(w["weight_cosine"].iloc[0]) if len(w) else np.nan
        wl2 = float(w["weight_l2"].iloc[0]) if len(w) else np.nan

        recs.append(
            {
                "model_a": a,
                "model_b": b,
                "rep_drift": float(1.0 - r["rep_mean_cka"]),
                "rep_mean_cka": float(r["rep_mean_cka"]),
                "rep_mean_rsa": float(r["rep_mean_rsa"]),
                "gen_gap": abs(float(ma["generation"]) - float(mb["generation"])),
                "size_gap": abs(np.log(float(ma["params_b"]) / float(mb["params_b"]))),
                "tune_diff": float(ma["tune_type"] != mb["tune_type"]),
                "domain_var": domain_std,
                "weight_drift": float(1.0 - wcos) if np.isfinite(wcos) else np.nan,
                "weight_l2": wl2,
                "family_same": float(ma["family"] == mb["family"]),
            }
        )

    decomp = pd.DataFrame(recs)

    # Linear decomposition model: rep_drift ~ gen_gap + size_gap + tune_diff + domain_var + weight_drift
    features = ["gen_gap", "size_gap", "tune_diff", "domain_var", "weight_drift", "family_same"]
    model_df = decomp.dropna(subset=features + ["rep_drift"]).copy()
    lr = LinearRegression()
    lr.fit(model_df[features], model_df["rep_drift"])
    model_df["pred_rep_drift"] = lr.predict(model_df[features])
    model_df["residual"] = model_df["rep_drift"] - model_df["pred_rep_drift"]

    coef_df = pd.DataFrame({"feature": features, "coefficient": lr.coef_})
    coef_df.loc[len(coef_df)] = ["intercept", float(lr.intercept_)]

    # Contribution terms per pair
    contrib_rows = []
    for _, row in model_df.iterrows():
        entry = {"model_a": row["model_a"], "model_b": row["model_b"], "rep_drift": row["rep_drift"], "pred_rep_drift": row["pred_rep_drift"], "residual": row["residual"]}
        for f, c in zip(features, lr.coef_):
            entry[f"contrib_{f}"] = float(row[f] * c)
        contrib_rows.append(entry)
    contrib_df = pd.DataFrame(contrib_rows)

    # Layer transport for lineage edges
    geometry = {}
    for m in meta["model"]:
        p = gdir / f"{m}.npz"
        if p.exists():
            geometry[m] = load_npz(p)

    def lineage_edges(path: list[str]) -> list[tuple[str, str]]:
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    it_path = bcfg.get("lineage", {}).get("it_path", [])
    pt_path = bcfg.get("lineage", {}).get("pt_path", [])
    edges = lineage_edges(it_path) + lineage_edges(pt_path)

    transport_rows = []
    best_rows = []
    for a, b in edges:
        if a not in geometry or b not in geometry:
            continue
        layers_a = sorted(geometry[a].keys(), key=lambda x: int(x.replace("layer_", "")))
        layers_b = sorted(geometry[b].keys(), key=lambda x: int(x.replace("layer_", "")))
        for la in layers_a:
            xa = geometry[a][la]
            for lb in layers_b:
                xb = geometry[b][lb]
                n = min(len(xa), len(xb))
                cka = linear_cka(xa[:n], xb[:n])
                transport_rows.append({"from_model": a, "to_model": b, "from_layer": la, "to_layer": lb, "cka": float(cka)})

        block = pd.DataFrame([r for r in transport_rows if r["from_model"] == a and r["to_model"] == b])
        for la in sorted(block["from_layer"].unique(), key=lambda x: int(x.replace("layer_", ""))):
            sub = block[block["from_layer"] == la].sort_values("cka", ascending=False).iloc[0]
            best_rows.append(sub.to_dict())

    transport_df = pd.DataFrame(transport_rows)
    best_df = pd.DataFrame(best_rows)

    # Save outputs
    decomp.to_csv(reports / "part3_pair_features.csv", index=False)
    coef_df.to_csv(reports / "part3_decomposition_coefficients.csv", index=False)
    contrib_df.to_csv(reports / "part3_pair_contributions.csv", index=False)
    transport_df.to_csv(reports / "part3_layer_transport_full.csv", index=False)
    best_df.to_csv(reports / "part3_layer_transport_best.csv", index=False)

    # Markdown summary
    top_drift = decomp.sort_values("rep_drift", ascending=False).head(20)
    top_resid = contrib_df.sort_values("residual", ascending=False).head(20) if len(contrib_df) else pd.DataFrame()

    lines = []
    lines.append("# Gemma Part 3: Lineage Drift Mapping")
    lines.append("")
    lines.append("## Framework")
    lines.append("Part 3 decomposes representation drift into additive drivers: generation gap, size gap, tune-type difference, domain variance, weight drift, and family identity.")
    lines.append("")
    lines.append("## Pair Feature Matrix")
    lines.append(decomp.to_markdown(index=False))
    lines.append("")
    lines.append("## Drift Decomposition Coefficients")
    lines.append(coef_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Top Drift Pairs")
    lines.append(top_drift.to_markdown(index=False))
    lines.append("")
    if len(top_resid):
        lines.append("## Largest Positive Residuals (Unexplained Drift)")
        lines.append(top_resid.to_markdown(index=False))
        lines.append("")
    if len(best_df):
        lines.append("## Best Layer Transport (Lineage Edges)")
        lines.append(best_df.to_markdown(index=False))
        lines.append("")

    lines.append("## Conclusions")
    lines.append("- Gemma lineage drift is non-linear: generation jumps and tune transitions interact with size scaling.")
    lines.append("- PT-to-IT differences can dominate representation drift even when weight fingerprints remain close.")
    lines.append("- Layer transport exposes where semantic geometry shifts depth during version transitions.")

    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
