from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build temporal drift dashboard for Gemma/LFM families.")
    p.add_argument("--reports-dir", default="outputs/reports_gemma_lfm_temporal_180")
    p.add_argument("--out-html", default="outputs/reports_gemma_lfm_temporal_180/temporal_drift_dashboard.html")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    r = Path(args.reports_dir)
    out = Path(args.out_html)
    out.parent.mkdir(parents=True, exist_ok=True)

    pair = pd.read_csv(r / "pairwise_similarity.csv")
    ptit = pd.read_csv(r / "gemma_pt_it_gap.csv") if (r / "gemma_pt_it_gap.csv").exists() else pd.DataFrame()
    traj = pd.read_csv(r / "gemma_temporal_trajectory.csv") if (r / "gemma_temporal_trajectory.csv").exists() else pd.DataFrame()
    joint = pd.read_csv(r / "rep_weight_joint_drift.csv") if (r / "rep_weight_joint_drift.csv").exists() else pd.DataFrame()

    pair_mean = pair.groupby(["model_a", "model_b"], as_index=False)["cka"].mean()
    pair_mean["pair"] = pair_mean["model_a"] + " vs " + pair_mean["model_b"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Pairwise Representation Alignment (Mean CKA)",
            "Gemma PT-vs-IT Gap",
            "Gemma Temporal IT Drift Path",
            "Joint Drift Space (Representation vs Weight)",
        ),
    )

    fig.add_trace(
        go.Bar(x=pair_mean["pair"], y=pair_mean["cka"], name="Mean CKA"),
        row=1,
        col=1,
    )

    if len(ptit):
        fig.add_trace(
            go.Bar(
                x=ptit["model_pt"] + " -> " + ptit["model_it"],
                y=ptit["rep_gap_1_minus_cka"],
                name="PT-IT Rep Gap",
            ),
            row=1,
            col=2,
        )

    if len(traj):
        fig.add_trace(
            go.Scatter(
                x=traj["to_generation"],
                y=traj["temporal_rep_drift_1_minus_cka"],
                mode="lines+markers+text",
                text=traj["to_model"],
                name="Temporal Rep Drift",
            ),
            row=2,
            col=1,
        )

    if len(joint):
        j = joint.copy()
        j["pair"] = j["model_a"] + " vs " + j["model_b"]
        fig.add_trace(
            go.Scatter(
                x=1.0 - j["rep_mean_cka"],
                y=1.0 - j["weight_cosine"],
                mode="markers+text",
                text=j["pair"],
                textposition="top center",
                name="Joint Drift",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(height=1000, width=1600, template="plotly_white", title="Gemma/LFM Temporal Drift Dashboard")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
