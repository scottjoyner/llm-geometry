from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open


LAYER_RE = re.compile(r"(?:layers|model\.layers|language_model\.layers)\.(\d+)\.")
KEEP_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layerwise weight-geometry comparison for Gemma 4B vs 27B.")
    p.add_argument("--models-dir", default="data/models")
    p.add_argument("--model-a", default="gemma3_4b_it")
    p.add_argument("--model-b", default="gemma3_27b_it")
    p.add_argument("--out-dir", default="outputs/reports_gemma_4b_27b_24")
    p.add_argument("--layer-stride", type=int, default=4)
    p.add_argument("--max-tensors-per-layer", type=int, default=24)
    return p.parse_args()


def collect_layer_vectors(model_path: Path, layer_stride: int, max_tensors_per_layer: int) -> dict[int, np.ndarray]:
    safes = sorted(model_path.glob("*.safetensors"))
    if not safes:
        raise FileNotFoundError(f"No safetensors files in {model_path}")

    layer_feats: dict[int, list[float]] = {}
    layer_counts: dict[int, int] = {}

    for sf in safes:
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                if not key.endswith(KEEP_SUFFIXES):
                    continue
                m = LAYER_RE.search(key)
                if not m:
                    continue
                layer = int(m.group(1))
                if layer % layer_stride != 0:
                    continue
                if layer_counts.get(layer, 0) >= max_tensors_per_layer:
                    continue

                arr = f.get_tensor(key).float().flatten().numpy()
                if arr.size == 0:
                    continue
                feats = [
                    float(arr.mean()),
                    float(arr.std()),
                    float(np.linalg.norm(arr) / np.sqrt(arr.size)),
                    float(np.mean(np.abs(arr))),
                    float(np.quantile(arr, 0.1)),
                    float(np.quantile(arr, 0.5)),
                    float(np.quantile(arr, 0.9)),
                ]
                layer_feats.setdefault(layer, []).extend(feats)
                layer_counts[layer] = layer_counts.get(layer, 0) + 1

    out: dict[int, np.ndarray] = {}
    for layer, feats in layer_feats.items():
        out[layer] = np.array(feats, dtype=np.float64)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = min(len(a), len(b))
    x = a[:d]
    y = b[:d]
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denom)


def map_layer(layer_a: int, max_a: int, max_b: int) -> int:
    if max_a <= 0:
        return 0
    frac = layer_a / max_a
    return int(round(frac * max_b))


def main() -> None:
    args = parse_args()
    model_a_path = Path(args.models_dir) / args.model_a
    model_b_path = Path(args.models_dir) / args.model_b
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a_layers = collect_layer_vectors(model_a_path, args.layer_stride, args.max_tensors_per_layer)
    b_layers = collect_layer_vectors(model_b_path, args.layer_stride, args.max_tensors_per_layer)
    if not a_layers or not b_layers:
        raise RuntimeError("Missing layer vectors for one or both models.")

    max_a = max(a_layers.keys())
    max_b = max(b_layers.keys())
    b_keys = sorted(b_layers.keys())

    rows = []
    for la in sorted(a_layers.keys()):
        mapped = map_layer(la, max_a, max_b)
        lb = min(b_keys, key=lambda x: abs(x - mapped))
        va = a_layers[la]
        vb = b_layers[lb]
        d = min(len(va), len(vb))
        l2 = float(np.linalg.norm(va[:d] - vb[:d]))
        rows.append(
            {
                "model_a": args.model_a,
                "model_b": args.model_b,
                "layer_a": la,
                "mapped_layer_b": lb,
                "feature_dim_used": d,
                "weight_geom_cosine": cosine(va, vb),
                "weight_geom_l2": l2,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "weight_layer_geometry.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "model_a": args.model_a,
                "model_b": args.model_b,
                "mean_weight_geom_cosine": float(df["weight_geom_cosine"].mean()),
                "mean_weight_geom_l2": float(df["weight_geom_l2"].mean()),
                "min_weight_geom_cosine": float(df["weight_geom_cosine"].min()),
                "max_weight_geom_cosine": float(df["weight_geom_cosine"].max()),
            }
        ]
    )
    summary.to_csv(out_dir / "weight_layer_geometry_summary.csv", index=False)
    print(f"Wrote {out_dir / 'weight_layer_geometry.csv'}")
    print(f"Wrote {out_dir / 'weight_layer_geometry_summary.csv'}")


if __name__ == "__main__":
    main()
