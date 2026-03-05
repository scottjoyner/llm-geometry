from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Part 4 Gemma drift cartography dashboard.")
    p.add_argument("--reports-dir", default="outputs/reports_gemma_part4")
    p.add_argument("--out-html", default="outputs/reports_gemma_part4/gemma_part4_dashboard.html")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reports = Path(args.reports_dir)
    out = Path(args.out_html)
    out.parent.mkdir(parents=True, exist_ok=True)

    emb = pd.read_csv(reports / "part4_model_embedding.csv")
    edges = pd.read_csv(reports / "part4_lineage_edges.csv")
    matrix = pd.read_csv(reports / "part4_pair_drift_matrix.csv")
    tune = pd.read_csv(reports / "part4_instruction_shift.csv")

    models = sorted(set(emb["model"].tolist()))
    heat = pd.DataFrame(0.0, index=models, columns=models)
    for _, row in matrix.iterrows():
        a = row["model_a"]
        b = row["model_b"]
        v = float(row["rep_drift"])
        heat.loc[a, b] = v
        heat.loc[b, a] = v

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scatter"}, {"type": "heatmap"}], [{"type": "bar"}, {"type": "scatter"}]],
        subplot_titles=(
            "Gemma Drift Trajectory (2D MDS)",
            "Gemma Pairwise Representation Drift",
            "PT to IT Shift by Generation",
            "Lineage Edge Drift: Representation vs Weight",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )

    # Trajectory points
    fig.add_trace(
        go.Scatter(
            x=emb["x"],
            y=emb["y"],
            mode="markers+text",
            text=emb["model"],
            textposition="top center",
            marker=dict(size=12, color=emb["generation"], colorscale="Viridis", showscale=True, colorbar=dict(title="gen")),
            name="models",
        ),
        row=1,
        col=1,
    )

    # Directed arrows for lineage/tuning edges
    emb_idx = emb.set_index("model")
    for _, r in edges.iterrows():
        x0 = float(emb_idx.loc[r["model_from"], "x"])
        y0 = float(emb_idx.loc[r["model_from"], "y"])
        x1 = float(emb_idx.loc[r["model_to"], "x"])
        y1 = float(emb_idx.loc[r["model_to"], "y"])
        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x1",
            yref="y1",
            axref="x1",
            ayref="y1",
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.0,
            arrowwidth=1.2,
            opacity=0.5,
        )

    fig.add_trace(
        go.Heatmap(
            z=heat.values,
            x=heat.columns.tolist(),
            y=heat.index.tolist(),
            colorscale="YlOrRd",
            colorbar=dict(title="rep_drift"),
            name="pairwise",
        ),
        row=1,
        col=2,
    )

    if len(tune):
        fig.add_trace(
            go.Bar(
                x=tune["generation"].astype(str),
                y=tune["rep_drift"],
                name="rep_drift",
                marker_color="#1f77b4",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=tune["generation"].astype(str),
                y=tune["drift_index"],
                name="drift_index",
                marker_color="#ff7f0e",
            ),
            row=2,
            col=1,
        )

    for etype, sub in edges.groupby("edge_type"):
        fig.add_trace(
            go.Scatter(
                x=sub["weight_drift"],
                y=sub["rep_drift"],
                mode="markers+text",
                text=sub["model_from"] + " -> " + sub["model_to"],
                textposition="top center",
                name=etype,
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title="Gemma Part 4 Drift Cartography Dashboard",
        template="plotly_white",
        width=1750,
        height=1150,
        barmode="group",
    )
    fig.update_xaxes(title_text="MDS X", row=1, col=1)
    fig.update_yaxes(title_text="MDS Y", row=1, col=1)
    fig.update_xaxes(title_text="generation", row=2, col=1)
    fig.update_yaxes(title_text="drift", row=2, col=1)
    fig.update_xaxes(title_text="weight_drift", row=2, col=2)
    fig.update_yaxes(title_text="rep_drift", row=2, col=2)

    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
