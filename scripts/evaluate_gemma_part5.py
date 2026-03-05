from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from llm_geometry.io_utils import load_yaml
from llm_geometry.metrics import linear_cka, procrustes_residual


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part 5 operator atlas for Gemma lineage drift.")
    p.add_argument("--models-config", default="configs/models_gemma_part3.yaml")
    p.add_argument("--analysis-config", default="configs/analysis_gemma_part5.yaml")
    p.add_argument("--geometry-dir", default="outputs/geometry_gemma_part3_360")
    p.add_argument("--prompt-csv", default="data/prompts_gemma_part3_360.csv")
    p.add_argument("--out-dir", default="outputs/reports_gemma_part5")
    p.add_argument("--out-md", default="docs/GEMMA_PART5_OPERATOR_ATLAS.md")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--ridge-lambda", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def layer_num(name: str) -> int:
    return int(name.replace("layer_", ""))


def align_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(x.shape[0], y.shape[0])
    d = min(x.shape[1], y.shape[1])
    return x[:n, :d], y[:n, :d]


def fit_ridge_operator(x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    d = x.shape[1]
    xtx = x.T @ x
    reg = ridge_lambda * np.eye(d, dtype=x.dtype)
    w = np.linalg.solve(xtx + reg, x.T @ y)
    return w


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.square(y_true - y_pred).sum())
    ss_tot = float(np.square(y_true).sum())
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def operator_stats(w: np.ndarray) -> tuple[np.ndarray, float, float]:
    s = np.linalg.svd(w, compute_uv=False)
    if s.sum() <= 1e-12:
        ent = 0.0
        pr = 0.0
    else:
        p = s / s.sum()
        ent = float(-(p * np.log(np.clip(p, 1e-12, None))).sum())
        pr = float((s.sum() ** 2) / np.square(s).sum())
    return s, ent, pr


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mcfg = load_yaml(args.models_config)
    acfg = load_yaml(args.analysis_config)["part5"]
    prompts = pd.read_csv(args.prompt_csv)
    if "id" not in prompts.columns:
        prompts["id"] = np.arange(len(prompts))
    if "domain" not in prompts.columns:
        prompts["domain"] = "unknown"

    meta = pd.DataFrame(mcfg["models"])
    meta = meta[["name", "family", "generation", "tune_type", "params_b"]].rename(columns={"name": "model"})
    meta_idx = meta.set_index("model")

    geometry: dict[str, dict[str, np.ndarray]] = {}
    for m in meta["model"].tolist():
        p = Path(args.geometry_dir) / f"{m}.npz"
        if p.exists():
            geometry[m] = load_npz(p)

    op_rows = []
    spec_rows = []
    transfer_rows = []
    prompt_rows = []
    edge_rows = []
    mode_rows = []

    for source, target, edge_type in acfg["edges"]:
        if source not in geometry or target not in geometry:
            continue
        layers_s = sorted(geometry[source].keys(), key=layer_num)
        layers_t = sorted(geometry[target].keys(), key=layer_num)
        common_layers = sorted(set(layers_s).intersection(layers_t), key=layer_num)
        if not common_layers:
            continue

        # Cross-layer transfer map via direct CKA
        for ls in layers_s:
            x = geometry[source][ls]
            best_t = None
            best_cka = -1.0
            for lt in layers_t:
                y = geometry[target][lt]
                xa, ya = align_xy(x, y)
                cka = float(linear_cka(xa, ya))
                transfer_rows.append(
                    {
                        "edge_type": edge_type,
                        "model_from": source,
                        "model_to": target,
                        "from_layer": ls,
                        "to_layer": lt,
                        "cka": cka,
                    }
                )
                if cka > best_cka:
                    best_cka = cka
                    best_t = lt
            transfer_rows.append(
                {
                    "edge_type": edge_type,
                    "model_from": source,
                    "model_to": target,
                    "from_layer": ls,
                    "to_layer": best_t,
                    "cka": best_cka,
                    "is_best": 1,
                    "layer_shift": layer_num(best_t) - layer_num(ls),
                }
            )

        layer_metrics = []
        for layer in common_layers:
            x, y = align_xy(geometry[source][layer], geometry[target][layer])
            x_c = x - x.mean(axis=0, keepdims=True)
            y_c = y - y.mean(axis=0, keepdims=True)

            n = len(x_c)
            idx = np.arange(n)
            rng.shuffle(idx)
            n_train = int(max(8, min(n - 8, round(n * args.train_frac))))
            train_idx = idx[:n_train]
            test_idx = idx[n_train:]
            x_train = x_c[train_idx]
            y_train = y_c[train_idx]
            x_test = x_c[test_idx]
            y_test = y_c[test_idx]

            w = fit_ridge_operator(x_train, y_train, ridge_lambda=float(args.ridge_lambda))
            y_pred_train = x_train @ w
            y_pred_test = x_test @ w

            cka_before = float(linear_cka(x_test, y_test))
            cka_after = float(linear_cka(y_pred_test, y_test))
            proc_before = float(procrustes_residual(x_test, y_test))
            proc_after = float(procrustes_residual(y_pred_test, y_test))
            op_r2 = r2_score(y_test, y_pred_test)
            op_r2_train = r2_score(y_train, y_pred_train)
            s, s_entropy, s_pr = operator_stats(w)

            rec = {
                "edge_type": edge_type,
                "model_from": source,
                "model_to": target,
                "layer": layer,
                "cka_before": cka_before,
                "cka_after": cka_after,
                "cka_gain": cka_after - cka_before,
                "procrustes_before": proc_before,
                "procrustes_after": proc_after,
                "procrustes_gain": proc_before - proc_after,
                "operator_r2_train": op_r2_train,
                "operator_r2": op_r2,
                "operator_fro": float(np.linalg.norm(w, ord="fro")),
                "operator_trace_mean": float(np.trace(w) / max(w.shape[0], 1)),
                "operator_spectral_entropy": s_entropy,
                "operator_spectral_pr": s_pr,
            }
            op_rows.append(rec)
            layer_metrics.append(rec)

            topk = int(acfg.get("spectrum_topk", 16))
            for i, sv in enumerate(s[:topk], start=1):
                spec_rows.append(
                    {
                        "edge_type": edge_type,
                        "model_from": source,
                        "model_to": target,
                        "layer": layer,
                        "mode_rank": i,
                        "singular_value": float(sv),
                    }
                )

            # Drift modes on prompt displacement
            delta = y_c - x_c
            u, sd, vt = np.linalg.svd(delta, full_matrices=False)
            k = min(int(acfg.get("top_modes", 5)), vt.shape[0])
            mode_axes = vt[:k, :]
            scores = delta @ mode_axes.T
            drift_norm = np.linalg.norm(delta, axis=1)

            prompt_n = min(len(scores), len(prompts))
            for idx in range(prompt_n):
                row = {
                    "edge_type": edge_type,
                    "model_from": source,
                    "model_to": target,
                    "layer": layer,
                    "prompt_id": int(prompts.iloc[idx]["id"]),
                    "domain": str(prompts.iloc[idx]["domain"]),
                    "drift_norm": float(drift_norm[idx]),
                }
                for m in range(k):
                    row[f"mode_{m+1}_score"] = float(scores[idx, m])
                prompt_rows.append(row)

            for m in range(k):
                mode_rows.append(
                    {
                        "edge_type": edge_type,
                        "model_from": source,
                        "model_to": target,
                        "layer": layer,
                        "mode_rank": m + 1,
                        "mode_sv_ratio": float(sd[m] / np.clip(sd.sum(), 1e-12, None)),
                    }
                )

        lm = pd.DataFrame(layer_metrics)
        edge_rows.append(
            {
                "edge_type": edge_type,
                "model_from": source,
                "model_to": target,
                "generation_from": float(meta_idx.loc[source, "generation"]),
                "generation_to": float(meta_idx.loc[target, "generation"]),
                "mean_operator_r2": float(lm["operator_r2"].mean()),
                "mean_cka_gain": float(lm["cka_gain"].mean()),
                "mean_procrustes_gain": float(lm["procrustes_gain"].mean()),
                "mean_spectral_pr": float(lm["operator_spectral_pr"].mean()),
                "mean_spectral_entropy": float(lm["operator_spectral_entropy"].mean()),
            }
        )

    op_df = pd.DataFrame(op_rows)
    spec_df = pd.DataFrame(spec_rows)
    transfer_df = pd.DataFrame(transfer_rows)
    prompt_df = pd.DataFrame(prompt_rows)
    edge_df = pd.DataFrame(edge_rows)
    mode_df = pd.DataFrame(mode_rows)

    # Save tables
    op_df.to_csv(out_dir / "part5_operator_metrics.csv", index=False)
    spec_df.to_csv(out_dir / "part5_operator_spectra.csv", index=False)
    transfer_df.to_csv(out_dir / "part5_layer_transfer_map.csv", index=False)
    prompt_df.to_csv(out_dir / "part5_prompt_drift.csv", index=False)
    edge_df.to_csv(out_dir / "part5_edge_summary.csv", index=False)
    mode_df.to_csv(out_dir / "part5_mode_summary.csv", index=False)

    # Domain summary at deepest shared layer for each edge
    domain_rows = []
    if len(prompt_df):
        for (a, b, et), g in prompt_df.groupby(["model_from", "model_to", "edge_type"]):
            deepest_layer = sorted(g["layer"].unique(), key=layer_num)[-1]
            gd = g[g["layer"] == deepest_layer]
            dsum = gd.groupby("domain", as_index=False).agg(
                mean_drift_norm=("drift_norm", "mean"),
                p90_drift_norm=("drift_norm", lambda x: float(np.quantile(x, 0.90))),
            )
            for _, r in dsum.iterrows():
                domain_rows.append(
                    {
                        "edge_type": et,
                        "model_from": a,
                        "model_to": b,
                        "layer": deepest_layer,
                        "domain": r["domain"],
                        "mean_drift_norm": float(r["mean_drift_norm"]),
                        "p90_drift_norm": float(r["p90_drift_norm"]),
                    }
                )
    domain_df = pd.DataFrame(domain_rows)
    domain_df.to_csv(out_dir / "part5_domain_drift.csv", index=False)

    # Markdown report
    lines = []
    lines.append("# Gemma Part 5: Operator Atlas")
    lines.append("")
    lines.append("## Framework")
    lines.append("Part 5 estimates layerwise linear drift operators W that map source prompt representations to target prompt representations.")
    lines.append("Operators are ridge-regularized and evaluated on held-out prompts (train/test split) to avoid trivial overfit in high dimensions.")
    lines.append("The report includes transport quality, operator spectra, layer transfer maps, and prompt-level drift modes.")
    lines.append("")
    lines.append("## Edge Summary")
    lines.append(edge_df.sort_values(["edge_type", "generation_from"]).to_markdown(index=False))
    lines.append("")
    if len(op_df):
        top_layers = op_df.sort_values("operator_r2", ascending=False).head(20)
        weak_layers = op_df.sort_values("operator_r2", ascending=True).head(20)
        lines.append("## Best Transport Layers")
        lines.append(top_layers.to_markdown(index=False))
        lines.append("")
        lines.append("## Weakest Transport Layers")
        lines.append(weak_layers.to_markdown(index=False))
        lines.append("")
    if len(domain_df):
        lines.append("## Domain Drift (Deepest Layer per Edge)")
        lines.append(domain_df.sort_values("mean_drift_norm", ascending=False).to_markdown(index=False))
        lines.append("")
    if len(mode_df):
        mode_top = mode_df[mode_df["mode_rank"] <= 3].copy()
        lines.append("## Drift Mode Energy (Top-3)")
        lines.append(mode_top.to_markdown(index=False))
        lines.append("")
    lines.append("## Conclusions")
    lines.append("- Operator fit quality identifies where drift is near-linear versus where non-linear shifts dominate.")
    lines.append("- Layer transfer maps show whether representational semantics migrate depth-wise during model evolution.")
    lines.append("- Prompt-level drift modes expose domain-specific stress points in the lineage transitions.")

    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
