from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from llm_geometry.io_utils import ensure_dir, load_yaml
from llm_geometry.metrics import (
    anisotropy,
    knn_overlap,
    linear_cka,
    participation_ratio,
    procrustes_residual,
    rsa_spearman,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multidimensional evaluation with bootstrap/permutation statistics.")
    p.add_argument("--benchmark-config", default="configs/benchmark_1200.yaml")
    p.add_argument("--geometry-dir", default="outputs/geometry")
    p.add_argument("--prompt-csv", default=None)
    p.add_argument("--out-dir", default="outputs/reports")
    p.add_argument("--bootstrap-sample-size", type=int, default=400)
    return p.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def metric_bundle(x: np.ndarray, y: np.ndarray, k: int) -> dict[str, float]:
    return {
        "cka": linear_cka(x, y),
        "rsa_spearman": rsa_spearman(x, y),
        "knn_overlap": knn_overlap(x, y, k=k),
        "procrustes_residual": procrustes_residual(x, y),
    }


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    fn,
    samples: int,
    seed: int,
    sample_size: int,
) -> tuple[float, float, float]:
    n = min(len(x), len(y))
    m = min(n, sample_size)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        idx = rng.integers(0, n, size=m)
        values.append(fn(x[idx], y[idx]))
    vals = np.array(values)
    return float(vals.mean()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def permutation_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    domain_idx: dict[str, np.ndarray],
    metric_fn,
    samples: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    keys = sorted(domain_idx)
    obs = [metric_fn(x[domain_idx[k]], y[domain_idx[k]]) for k in keys]
    obs_var = float(np.var(obs))

    all_idx = np.arange(min(len(x), len(y)))
    sizes = [len(domain_idx[k]) for k in keys]
    null_vars = []
    for _ in range(samples):
        perm = rng.permutation(all_idx)
        start = 0
        vals = []
        for s in sizes:
            idx = perm[start : start + s]
            start += s
            vals.append(metric_fn(x[idx], y[idx]))
        null_vars.append(float(np.var(vals)))

    null_arr = np.array(null_vars)
    return float((1 + np.sum(null_arr >= obs_var)) / (1 + len(null_arr)))


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.benchmark_config)
    eval_cfg = cfg["evaluation"]

    prompt_csv = Path(args.prompt_csv or cfg["benchmark"]["prompt_csv"])
    prompts = pd.read_csv(prompt_csv)
    out_dir = ensure_dir(args.out_dir)

    files = sorted(Path(args.geometry_dir).glob("*.npz"))
    if len(files) < 2:
        raise SystemExit("Need at least two geometry artifacts.")

    geometry = {f.stem: load_npz(f) for f in files}
    n_common = min(min(v.shape[0] for v in g.values()) for g in geometry.values())
    prompts = prompts.iloc[:n_common].reset_index(drop=True)
    domains = sorted(prompts["domain"].unique())
    domain_idx = {d: np.where(prompts["domain"].values == d)[0] for d in domains}

    summary_rows = []
    pair_rows = []
    domain_rows = []
    ci_rows = []
    perm_rows = []

    for model_name, layers in geometry.items():
        for layer_name, vecs in layers.items():
            vecs = vecs[:n_common]
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

    boot_n = int(eval_cfg.get("bootstrap_samples", 40))
    perm_n = int(eval_cfg.get("permutation_samples", 20))
    knn_k = int(eval_cfg.get("knn_k", 10))

    for (m1, g1), (m2, g2) in combinations(geometry.items(), 2):
        for layer in sorted(set(g1).intersection(g2)):
            x = g1[layer][:n_common]
            y = g2[layer][:n_common]

            metrics = metric_bundle(x, y, k=knn_k)
            pair_rows.append({"model_a": m1, "model_b": m2, "layer": layer, **metrics})

            for metric_name, fn in [("cka", linear_cka), ("rsa_spearman", rsa_spearman)]:
                mean, low, high = bootstrap_ci(
                    x,
                    y,
                    fn=fn,
                    samples=boot_n,
                    seed=20260303,
                    sample_size=args.bootstrap_sample_size,
                )
                ci_rows.append(
                    {
                        "model_a": m1,
                        "model_b": m2,
                        "layer": layer,
                        "metric": metric_name,
                        "boot_mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "n_boot": boot_n,
                        "boot_sample_size": min(args.bootstrap_sample_size, n_common),
                    }
                )

            for d in domains:
                idx = domain_idx[d]
                dm = metric_bundle(x[idx], y[idx], k=min(knn_k, max(len(idx) - 1, 1)))
                domain_rows.append({"model_a": m1, "model_b": m2, "layer": layer, "domain": d, **dm})

            pval = permutation_pvalue(x, y, domain_idx, linear_cka, perm_n, seed=20260303)
            perm_rows.append(
                {
                    "model_a": m1,
                    "model_b": m2,
                    "layer": layer,
                    "metric": "cka_domain_variance",
                    "p_value": pval,
                    "n_perm": perm_n,
                }
            )

    pd.DataFrame(summary_rows).sort_values(["model", "layer"]).to_csv(out_dir / "geometry_summary.csv", index=False)
    pd.DataFrame(pair_rows).sort_values(["model_a", "model_b", "layer"]).to_csv(
        out_dir / "pairwise_similarity.csv", index=False
    )
    pd.DataFrame(domain_rows).sort_values(["model_a", "model_b", "layer", "domain"]).to_csv(
        out_dir / "domain_similarity.csv", index=False
    )
    pd.DataFrame(ci_rows).sort_values(["model_a", "model_b", "layer", "metric"]).to_csv(
        out_dir / "pairwise_bootstrap_ci.csv", index=False
    )
    pd.DataFrame(perm_rows).sort_values(["model_a", "model_b", "layer"]).to_csv(
        out_dir / "permutation_tests.csv", index=False
    )

    print(f"Wrote reports in {out_dir}")


if __name__ == "__main__":
    main()
