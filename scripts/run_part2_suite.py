from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Part 2 benchmark suite for requested model set.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--models-config", default="configs/models_part2.yaml")
    p.add_argument("--benchmark-config", default="configs/benchmark_part2_360.yaml")
    p.add_argument("--prompts-csv", default="data/prompts_part2_360.csv")
    p.add_argument("--models-dir", default="data/models")
    p.add_argument("--geometry-dir", default="outputs/geometry_part2_360")
    p.add_argument("--reports-dir", default="outputs/reports_part2_360")
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
        "240",
    ])

    run([
        args.python,
        "scripts/build_geometry_atlas.py",
        "--geometry-dir",
        args.geometry_dir,
        "--reports-dir",
        args.reports_dir,
        "--prompt-csv",
        args.prompts_csv,
        "--out-html",
        f"{args.reports_dir}/geometry_atlas_part2.html",
        "--layer",
        "layer_12",
    ])

    run([
        args.python,
        "scripts/analyze_part2_drift.py",
        "--models-config",
        args.models_config,
        "--reports-dir",
        args.reports_dir,
        "--out-md",
        "docs/PART2_DRIFT_REPORT.md",
    ])


if __name__ == "__main__":
    main()
