from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from llm_geometry.io_utils import ensure_dir, load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a deterministic 1200-prompt benchmark with domain labels.")
    p.add_argument("--config", default="configs/benchmark_1200.yaml")
    p.add_argument("--out-csv", default=None)
    return p.parse_args()


def make_prompts(domain: str, n: int, rng: random.Random) -> list[str]:
    objects = [
        "vector field", "manifold", "embedding", "tensor", "graph", "trajectory", "polytope", "basis",
        "operator", "latent map", "coordinate frame", "geodesic", "kernel", "distance matrix", "projection"
    ]
    actions = [
        "compare", "transform", "stabilize", "optimize", "decompose", "align", "perturb", "normalize",
        "project", "cluster", "interpolate", "factorize"
    ]
    constraints = [
        "under rotational invariance", "with bounded curvature", "under noisy measurements",
        "with sparse observations", "under fixed token budget", "across two scales",
        "while preserving neighborhood topology", "with contrastive supervision"
    ]

    prompts: list[str] = []
    for i in range(n):
        a = rng.choice(actions)
        o1 = rng.choice(objects)
        o2 = rng.choice(objects)
        c = rng.choice(constraints)

        if domain == "logical_reasoning":
            prompts.append(
                f"Given premises about a {o1}, derive a contradiction test and {a} the conclusion for a {o2} {c}."
            )
        elif domain == "algebraic_geometry":
            prompts.append(
                f"Compute an explicit mapping from a {o1} to a {o2}, then {a} basis vectors and discuss rank behavior {c}."
            )
        elif domain == "analogy_composition":
            prompts.append(
                f"Complete the analogy: '{o1} is to local structure as {o2} is to global structure'; then {a} the relation {c}."
            )
        elif domain == "factual_recall":
            prompts.append(
                f"State concise facts about how a {o1} is used in machine learning, then {a} with a {o2} example {c}."
            )
        elif domain == "narrative_structure":
            prompts.append(
                f"Write a short technical narrative where a researcher must {a} a {o1} and a {o2} {c}, ending with a tradeoff."
            )
        elif domain == "symbolic_code":
            prompts.append(
                f"Provide pseudocode to {a} a {o1} into a {o2} {c}; include complexity analysis and failure modes."
            )
        else:
            raise ValueError(f"Unknown domain: {domain}")

        prompts[-1] += f" [benchmark_id={domain}_{i:03d}]"

    return prompts


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    bcfg = cfg["benchmark"]

    seed = int(bcfg.get("seed", 20260303))
    rng = random.Random(seed)

    domains = list(bcfg["domains"])
    per_domain = int(bcfg["prompts_per_domain"])
    out_csv = Path(args.out_csv or bcfg["prompt_csv"])
    ensure_dir(out_csv.parent)

    rows = []
    idx = 0
    for domain in domains:
        prompts = make_prompts(domain, per_domain, rng)
        for p in prompts:
            rows.append({"id": idx, "domain": domain, "prompt": p})
            idx += 1

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "domain", "prompt"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} prompts to {out_csv}")


if __name__ == "__main__":
    main()
