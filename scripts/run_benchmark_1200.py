from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full 1200-prompt geometry benchmark and visualization pipeline.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--benchmark-config", default="configs/benchmark_1200.yaml")
    p.add_argument("--prompts-csv", default="data/prompts_1200.csv")
    p.add_argument("--skip-download", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run([args.python, "scripts/generate_benchmark_prompts.py", "--config", args.benchmark_config, "--out-csv", args.prompts_csv])

    if not args.skip_download:
        run([args.python, "scripts/download_models.py"])

    run([
        args.python,
        "scripts/extract_geometry.py",
        "--prompts",
        args.prompts_csv,
    ])

    run([
        args.python,
        "scripts/evaluate_multidim.py",
        "--benchmark-config",
        args.benchmark_config,
        "--prompt-csv",
        args.prompts_csv,
    ])

    run([
        args.python,
        "scripts/build_geometry_atlas.py",
        "--prompt-csv",
        args.prompts_csv,
    ])


if __name__ == "__main__":
    main()
