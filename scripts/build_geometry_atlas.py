from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from llm_geometry.io_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Geometry Atlas visualization HTML.")
    p.add_argument("--geometry-dir", default="outputs/geometry")
    p.add_argument("--reports-dir", default="outputs/reports")
    p.add_argument("--prompt-csv", default="data/prompts_1200.csv")
    p.add_argument("--out-html", default="outputs/reports/geometry_atlas.html")
    p.add_argument("--layer", default="layer_12")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gdir = Path(args.geometry_dir)
    rdir = Path(args.reports_dir)
    out_path = Path(args.out_html)
    ensure_dir(out_path.parent)

    summary = pd.read_csv(rdir / "geometry_summary.csv")
    pairs = pd.read_csv(rdir / "pairwise_similarity.csv")
    domain = pd.read_csv(rdir / "domain_similarity.csv")
    prompts = pd.read_csv(args.prompt_csv)

    npzs = sorted(gdir.glob("*.npz"))
    model_vectors = []
    model_names = []
    nmin = None
    for f in npzs:
        d = np.load(f)
        if args.layer not in d.files:
            continue
        v = d[args.layer]
        nmin = len(v) if nmin is None else min(nmin, len(v))
        model_vectors.append(v)
        model_names.append(f.stem)

    if not model_vectors:
        raise SystemExit(f"No geometry vectors found for layer {args.layer}")

    nmin = int(nmin)
    prompts = prompts.iloc[:nmin].reset_index(drop=True)

    concat = []
    tags_model = []
    tags_domain = []
    for name, vecs in zip(model_names, model_vectors):
        vv = vecs[:nmin]
        pca = PCA(n_components=3, random_state=42)
        emb = pca.fit_transform(vv)
        concat.append(emb)
        tags_model.extend([name] * nmin)
        tags_domain.extend(prompts["domain"].tolist())

    xyz = np.vstack(concat)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "heatmap"}], [{"type": "polar"}, {"type": "heatmap"}]],
        subplot_titles=(
            "Geometry Atlas 3D (Layer Projection)",
            "Alignment River (Pairwise CKA by Layer)",
            "Model Geometry Fingerprints",
            "Domain-Tension Matrix (CKA by Domain)",
        ),
    )

    unique_models = sorted(set(tags_model))
    for m in unique_models:
        idx = [i for i, name in enumerate(tags_model) if name == m]
        fig.add_trace(
            go.Scatter3d(
                x=xyz[idx, 0],
                y=xyz[idx, 1],
                z=xyz[idx, 2],
                mode="markers",
                name=m,
                marker={"size": 2},
                text=[tags_domain[i] for i in idx],
                hovertemplate="model=%{name}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<br>domain=%{text}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    cka_pivot = pairs.pivot_table(index=["model_a", "model_b"], columns="layer", values="cka", aggfunc="mean")
    fig.add_trace(
        go.Heatmap(
            z=cka_pivot.values,
            x=[str(c) for c in cka_pivot.columns],
            y=[f"{a} vs {b}" for a, b in cka_pivot.index],
            colorbar={"title": "CKA"},
            colorscale="Viridis",
            name="CKA Heatmap",
        ),
        row=1,
        col=2,
    )

    fp = summary.copy()
    fp["layer_id"] = fp["layer"].str.replace("layer_", "", regex=False).astype(int)
    fp = fp.sort_values(["model", "layer_id"])
    for m in sorted(fp["model"].unique()):
        sub = fp[fp["model"] == m]
        fig.add_trace(
            go.Scatterpolar(
                r=sub["participation_ratio"],
                theta=sub["layer"],
                mode="lines+markers",
                name=f"{m} PR",
                legendgroup=m,
            ),
            row=2,
            col=1,
        )

    d_pivot = domain.pivot_table(index=["model_a", "model_b"], columns="domain", values="cka", aggfunc="mean")
    fig.add_trace(
        go.Heatmap(
            z=d_pivot.values,
            x=[str(c) for c in d_pivot.columns],
            y=[f"{a} vs {b}" for a, b in d_pivot.index],
            colorbar={"title": "Domain CKA"},
            colorscale="Cividis",
            name="Domain Heatmap",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="Geometric Atlas: Multidimensional Representation Alignment",
        height=1200,
        width=1800,
        template="plotly_white",
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
