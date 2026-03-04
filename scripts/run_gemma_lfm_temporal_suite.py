from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Gemma/LFM temporal drift suite.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--models-config", default="configs/models_gemma_lfm_temporal.yaml")
    p.add_argument("--benchmark-config", default="configs/benchmark_gemma_lfm_temporal_180.yaml")
    p.add_argument("--prompts-csv", default="data/prompts_gemma_lfm_temporal_180.csv")
    p.add_argument("--models-dir", default="data/models")
    p.add_argument("--geometry-dir", default="outputs/geometry_gemma_lfm_temporal_180")
    p.add_argument("--reports-dir", default="outputs/reports_gemma_lfm_temporal_180")
    p.add_argument("--skip-download", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run([
        args.python,
        "scripts/generate_benchmark_prompts.py",
        "--config",
        args.benchmark_config,
        "--out-csv",
        args.prompts_csv,
    ])

    if not args.skip_download:
        run([
            args.python,
            "scripts/download_models.py",
            "--config",
            args.models_config,
            "--out-dir",
            args.models_dir,
        ])

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
        "180",
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
        "docs/GEMMA_LFM_TEMPORAL_OBSERVATIONS.md",
    ])

    run([
        args.python,
        "scripts/build_temporal_drift_dashboard.py",
        "--reports-dir",
        args.reports_dir,
        "--out-html",
        f"{args.reports_dir}/temporal_drift_dashboard.html",
    ])


if __name__ == "__main__":
    main()
