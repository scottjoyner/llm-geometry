from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from llm_geometry.io_utils import ensure_dir
from llm_geometry.metrics import anisotropy, linear_cka, participation_ratio, rsa_spearman


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare model geometry artifacts.")
    p.add_argument("--geometry-dir", default="outputs/geometry")
    p.add_argument("--out-dir", default="outputs/reports")
    return p.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def main() -> None:
    args = parse_args()
    gdir = Path(args.geometry_dir)
    out_dir = ensure_dir(args.out_dir)

    files = sorted(gdir.glob("*.npz"))
    if len(files) < 2:
        raise SystemExit("Need at least two geometry files for comparison.")

    geometry = {f.stem: load_npz(f) for f in files}

    summary_rows = []
    pair_rows = []

    for model_name, layers in geometry.items():
        for layer_name, vecs in layers.items():
            summary_rows.append(
                {
                    "model": model_name,
                    "layer": layer_name,
                    "n": int(vecs.shape[0]),
                    "d": int(vecs.shape[1]),
                    "participation_ratio": participation_ratio(vecs),
                    "anisotropy": anisotropy(vecs),
                }
            )

    for (m1, g1), (m2, g2) in combinations(geometry.items(), 2):
        shared_layers = sorted(set(g1).intersection(g2))
        for layer in shared_layers:
            x = g1[layer]
            y = g2[layer]
            n = min(len(x), len(y))
            if n < 4:
                continue
            x = x[:n]
            y = y[:n]
            pair_rows.append(
                {
                    "model_a": m1,
                    "model_b": m2,
                    "layer": layer,
                    "n": n,
                    "cka": linear_cka(x, y),
                    "rsa_spearman": rsa_spearman(x, y),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["model", "layer"])
    pairs_df = pd.DataFrame(pair_rows).sort_values(["model_a", "model_b", "layer"])

    summary_path = out_dir / "geometry_summary.csv"
    pair_path = out_dir / "pairwise_similarity.csv"
    summary_df.to_csv(summary_path, index=False)
    pairs_df.to_csv(pair_path, index=False)

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {pair_path}")


if __name__ == "__main__":
    main()
