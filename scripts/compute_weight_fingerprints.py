from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open
from sklearn.metrics.pairwise import cosine_similarity

from llm_geometry.io_utils import ensure_dir, load_yaml


LAYER_RE = re.compile(r"(?:layers|model\.layers|transformer\.h)\.(\d+)\.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute lightweight weight-fingerprint drift metrics.")
    p.add_argument("--models-config", default="configs/models_gemma_lfm_temporal.yaml")
    p.add_argument("--models-dir", default="data/models")
    p.add_argument("--out-dir", default="outputs/reports_gemma_lfm_temporal_180")
    p.add_argument("--layer-stride", type=int, default=4)
    p.add_argument("--max-tensors", type=int, default=240)
    return p.parse_args()


def tensor_selected(key: str, layer_stride: int) -> bool:
    keep_suffixes = (
        "embed_tokens.weight",
        "lm_head.weight",
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
        "norm.weight",
        "layernorm.weight",
    )
    if key.endswith(keep_suffixes):
        m = LAYER_RE.search(key)
        if m:
            layer = int(m.group(1))
            return layer % layer_stride == 0
        return True
    return False


def fingerprint_model(model_path: Path, layer_stride: int, max_tensors: int) -> tuple[np.ndarray, dict[str, float]]:
    safes = sorted(model_path.glob("*.safetensors"))
    if not safes:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    feats: list[float] = []
    agg_sum = 0.0
    agg_sq = 0.0
    agg_abs = 0.0
    agg_n = 0
    used = 0

    for sf in safes:
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                if not tensor_selected(key, layer_stride):
                    continue
                t = f.get_tensor(key).float()
                arr = t.flatten().numpy()
                n = arr.size
                if n == 0:
                    continue

                mean = float(arr.mean())
                std = float(arr.std())
                l2 = float(np.linalg.norm(arr) / np.sqrt(n))
                absmean = float(np.mean(np.abs(arr)))

                feats.extend([mean, std, l2, absmean])

                agg_sum += float(arr.sum())
                agg_sq += float(np.square(arr).sum())
                agg_abs += float(np.abs(arr).sum())
                agg_n += int(n)

                used += 1
                if used >= max_tensors:
                    break
        if used >= max_tensors:
            break

    if agg_n == 0:
        raise RuntimeError(f"No tensors selected from {model_path}")

    global_mean = agg_sum / agg_n
    global_rms = np.sqrt(agg_sq / agg_n)
    global_abs = agg_abs / agg_n

    # Fixed-length projection of variable feature vector.
    proj_dim = 128
    rng = np.random.default_rng(20260304)
    proj = rng.normal(0.0, 1.0 / np.sqrt(max(len(feats), 1)), size=(proj_dim, max(len(feats), 1)))
    vec = np.array(feats if feats else [0.0], dtype=np.float64)
    fp = proj @ vec

    stats = {
        "tensors_used": float(used),
        "global_mean": float(global_mean),
        "global_rms": float(global_rms),
        "global_abs_mean": float(global_abs),
        "global_n": float(agg_n),
    }
    return fp.astype(np.float64), stats


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.models_config)
    models_dir = Path(args.models_dir)
    out_dir = ensure_dir(args.out_dir)

    rows = []
    fps = {}

    for m in cfg["models"]:
        name = m["name"]
        mp = models_dir / name
        if not mp.exists():
            print(f"Skip {name}: missing {mp}")
            continue
        try:
            fp, stats = fingerprint_model(mp, layer_stride=args.layer_stride, max_tensors=args.max_tensors)
            fps[name] = fp
            rows.append(
                {
                    "model": name,
                    "family": m.get("family", "unknown"),
                    "generation": float(m.get("generation", 0.0)),
                    "tune_type": m.get("tune_type", "unknown"),
                    "params_b": float(m.get("params_b", 0.0)),
                    **stats,
                }
            )
            print(f"Fingerprint OK: {name} | tensors={int(stats['tensors_used'])}")
        except Exception as e:
            print(f"Fingerprint failed: {name} | {e}")

    stats_df = pd.DataFrame(rows)
    stats_path = out_dir / "weight_fingerprint_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    pair_rows = []
    names = sorted(fps)
    if len(names) >= 2:
        mat = np.vstack([fps[n] for n in names])
        cos = cosine_similarity(mat)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if j <= i:
                    continue
                da = fps[a]
                db = fps[b]
                l2 = float(np.linalg.norm(da - db))
                pair_rows.append(
                    {
                        "model_a": a,
                        "model_b": b,
                        "weight_cosine": float(cos[i, j]),
                        "weight_l2": l2,
                    }
                )

    pair_df = pd.DataFrame(pair_rows)
    pair_path = out_dir / "weight_pairwise_drift.csv"
    pair_df.to_csv(pair_path, index=False)

    fp_json = out_dir / "weight_fingerprint_vectors.json"
    fp_json.write_text(json.dumps({k: fps[k].tolist() for k in names}), encoding="utf-8")

    print(f"Wrote {stats_path}")
    print(f"Wrote {pair_path}")
    print(f"Wrote {fp_json}")


if __name__ == "__main__":
    main()
