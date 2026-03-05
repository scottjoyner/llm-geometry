from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Part 3 Gemma lineage suite.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--models-config", default="configs/models_gemma_part3.yaml")
    p.add_argument("--benchmark-config", default="configs/benchmark_gemma_part3_360.yaml")
    p.add_argument("--prompts-csv", default="data/prompts_gemma_part3_360.csv")
    p.add_argument("--models-dir", default="data/models")
    p.add_argument("--geometry-dir", default="outputs/geometry_gemma_part3_360")
    p.add_argument("--reports-dir", default="outputs/reports_gemma_part3_360")
    p.add_argument("--skip-download", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run([args.python, "scripts/generate_benchmark_prompts.py", "--config", args.benchmark_config, "--out-csv", args.prompts_csv])

    if not args.skip_download:
        run([args.python, "scripts/download_models.py", "--config", args.models_config, "--out-dir", args.models_dir])

    run([
        args.python,
        "scripts/extract_geometry.py",
        "--config",
        args.models_config,
        "--models-dir",
        args.models_dir,
        "--prompts",
        args.prompts_csv,
        "--out-dir",
        args.geometry_dir,
    ])

    run([
        args.python,
        "scripts/evaluate_multidim.py",
        "--benchmark-config",
        args.benchmark_config,
        "--geometry-dir",
        args.geometry_dir,
        "--prompt-csv",
        args.prompts_csv,
        "--out-dir",
        args.reports_dir,
        "--bootstrap-sample-size",
        "240",
    ])

    run([
        args.python,
        "scripts/compute_weight_fingerprints.py",
        "--models-config",
        args.models_config,
        "--models-dir",
        args.models_dir,
        "--out-dir",
        args.reports_dir,
        "--layer-stride",
        "2",
        "--max-tensors",
        "260",
    ])

    run([
        args.python,
        "scripts/evaluate_gemma_lfm_temporal.py",
        "--models-config",
        args.models_config,
        "--reports-dir",
        args.reports_dir,
        "--out-dir",
        args.reports_dir,
        "--out-md",
        "docs/GEMMA_PART3_TEMPORAL_BASELINE.md",
    ])

    run([
        args.python,
        "scripts/evaluate_gemma_part3.py",
        "--models-config",
        args.models_config,
        "--benchmark-config",
        args.benchmark_config,
        "--geometry-dir",
        args.geometry_dir,
        "--reports-dir",
        args.reports_dir,
        "--out-md",
        "docs/GEMMA_PART3_OBSERVATIONS.md",
    ])

    run([
        args.python,
        "scripts/build_gemma_part3_dashboard.py",
        "--reports-dir",
        args.reports_dir,
        "--out-html",
        f"{args.reports_dir}/gemma_part3_dashboard.html",
    ])


if __name__ == "__main__":
    main()
