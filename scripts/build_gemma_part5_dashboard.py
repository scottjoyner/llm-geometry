from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Part 5 Gemma operator atlas dashboard.")
    p.add_argument("--reports-dir", default="outputs/reports_gemma_part5")
    p.add_argument("--out-html", default="outputs/reports_gemma_part5/gemma_part5_dashboard.html")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reports = Path(args.reports_dir)
    out = Path(args.out_html)
    out.parent.mkdir(parents=True, exist_ok=True)

    op = pd.read_csv(reports / "part5_operator_metrics.csv")
    edges = pd.read_csv(reports / "part5_edge_summary.csv")
    transfer = pd.read_csv(reports / "part5_layer_transfer_map.csv")
    domain = pd.read_csv(reports / "part5_domain_drift.csv")

    op["edge_label"] = op["model_from"] + " -> " + op["model_to"] + " (" + op["edge_type"] + ")"
    edges["edge_label"] = edges["model_from"] + " -> " + edges["model_to"] + " (" + edges["edge_type"] + ")"
    transfer["edge_label"] = transfer["model_from"] + " -> " + transfer["model_to"] + " (" + transfer["edge_type"] + ")"

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "heatmap"}, {"type": "bar"}], [{"type": "scatter"}, {"type": "bar"}]],
        subplot_titles=(
            "Operator R2 by Edge and Layer",
            "Edge-Level Mean Operator Quality",
            "Layer Transfer Shift (Best Mapping)",
            "Deep-Layer Domain Drift",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )

    pivot = op.pivot_table(index="edge_label", columns="layer", values="operator_r2", aggfunc="mean")
    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Viridis",
            colorbar=dict(title="R2"),
            name="operator_r2",
        ),
        row=1,
        col=1,
    )

    edges_plot = edges.sort_values("mean_operator_r2", ascending=False)
    fig.add_trace(
        go.Bar(
            x=edges_plot["edge_label"],
            y=edges_plot["mean_operator_r2"],
            marker_color="#1f77b4",
            name="mean_operator_r2",
        ),
        row=1,
        col=2,
    )

    best = transfer[transfer.get("is_best", 0) == 1].copy()
    if len(best):
        fig.add_trace(
            go.Scatter(
                x=best["from_layer"],
                y=best["layer_shift"],
                mode="markers+text",
                text=best["edge_label"],
                textposition="top center",
                marker=dict(size=10, color=best["cka"], colorscale="YlOrRd", showscale=True, colorbar=dict(title="CKA")),
                name="layer_shift",
            ),
            row=2,
            col=1,
        )

    if len(domain):
        domain["edge_label"] = domain["model_from"] + " -> " + domain["model_to"] + " (" + domain["edge_type"] + ")"
        dom = domain.sort_values("mean_drift_norm", ascending=False).head(30)
        fig.add_trace(
            go.Bar(
                x=dom["domain"] + " | " + dom["edge_label"],
                y=dom["mean_drift_norm"],
                marker_color="#ff7f0e",
                name="domain_drift",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title="Gemma Part 5 Operator Atlas",
        template="plotly_white",
        width=1800,
        height=1200,
    )
    fig.update_xaxes(title_text="layer", row=1, col=1)
    fig.update_yaxes(title_text="edge", row=1, col=1)
    fig.update_xaxes(title_text="edge", row=1, col=2)
    fig.update_yaxes(title_text="mean_operator_r2", row=1, col=2)
    fig.update_xaxes(title_text="source layer", row=2, col=1)
    fig.update_yaxes(title_text="best target shift", row=2, col=1)
    fig.update_xaxes(title_text="domain | edge", row=2, col=2)
    fig.update_yaxes(title_text="mean drift norm", row=2, col=2)

    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
