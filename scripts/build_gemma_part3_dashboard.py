from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Part 3 Gemma lineage dashboard.")
    p.add_argument("--reports-dir", default="outputs/reports_gemma_part3_360")
    p.add_argument("--out-html", default="outputs/reports_gemma_part3_360/gemma_part3_dashboard.html")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    r = Path(args.reports_dir)
    out = Path(args.out_html)
    out.parent.mkdir(parents=True, exist_ok=True)

    pair = pd.read_csv(r / "part3_pair_features.csv")
    coef = pd.read_csv(r / "part3_decomposition_coefficients.csv")
    best = pd.read_csv(r / "part3_layer_transport_best.csv")

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scatter"}, {"type": "bar"}], [{"type": "scatter"}, {"type": "domain"}]],
        subplot_titles=(
            "Rep Drift vs Weight Drift",
            "Decomposition Coefficients",
            "Generation Gap vs Rep Drift",
            "Layer Transport Sankey",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=pair["weight_drift"],
            y=pair["rep_drift"],
            mode="markers+text",
            text=pair["model_a"] + " vs " + pair["model_b"],
            textposition="top center",
            name="pairs",
        ),
        row=1,
        col=1,
    )

    coef_plot = coef[coef["feature"] != "intercept"]
    fig.add_trace(
        go.Bar(x=coef_plot["feature"], y=coef_plot["coefficient"], name="coef"),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=pair["gen_gap"],
            y=pair["rep_drift"],
            mode="markers",
            text=pair["model_a"] + " vs " + pair["model_b"],
            name="gen-vs-drift",
        ),
        row=2,
        col=1,
    )

    # Sankey from first edge available in best transport
    if len(best):
        edge = best[["from_model", "to_model"]].drop_duplicates().iloc[0]
        s = best[(best["from_model"] == edge["from_model"]) & (best["to_model"] == edge["to_model"])].copy()
        from_layers = sorted(s["from_layer"].unique(), key=lambda x: int(x.replace("layer_", "")))
        to_layers = sorted(s["to_layer"].unique(), key=lambda x: int(x.replace("layer_", "")))
        labels = [f"{edge['from_model']}:{l}" for l in from_layers] + [f"{edge['to_model']}:{l}" for l in to_layers]
        idx = {lab: i for i, lab in enumerate(labels)}

        src = [idx[f"{edge['from_model']}:{r['from_layer']}"] for _, r in s.iterrows()]
        tgt = [idx[f"{edge['to_model']}:{r['to_layer']}"] for _, r in s.iterrows()]
        val = [float(r["cka"]) for _, r in s.iterrows()]

        sankey = go.Sankey(
            node=dict(label=labels, pad=15, thickness=12),
            link=dict(source=src, target=tgt, value=val),
            arrangement="snap",
            name="transport",
        )
        fig.add_trace(sankey, row=2, col=2)

    fig.update_layout(height=1100, width=1700, template="plotly_white", title="Gemma Part 3 Drift Dashboard")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
