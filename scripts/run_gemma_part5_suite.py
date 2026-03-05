from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Part 5 Gemma operator atlas suite.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--models-config", default="configs/models_gemma_part3.yaml")
    p.add_argument("--analysis-config", default="configs/analysis_gemma_part5.yaml")
    p.add_argument("--geometry-dir", default="outputs/geometry_gemma_part3_360")
    p.add_argument("--prompt-csv", default="data/prompts_gemma_part3_360.csv")
    p.add_argument("--reports-output", default="outputs/reports_gemma_part5")
    p.add_argument("--out-md", default="docs/GEMMA_PART5_OPERATOR_ATLAS.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        [
            args.python,
            "scripts/evaluate_gemma_part5.py",
            "--models-config",
            args.models_config,
            "--analysis-config",
            args.analysis_config,
            "--geometry-dir",
            args.geometry_dir,
            "--prompt-csv",
            args.prompt_csv,
            "--out-dir",
            args.reports_output,
            "--out-md",
            args.out_md,
        ]
    )
    run(
        [
            args.python,
            "scripts/build_gemma_part5_dashboard.py",
            "--reports-dir",
            args.reports_output,
            "--out-html",
            f"{args.reports_output}/gemma_part5_dashboard.html",
        ]
    )


if __name__ == "__main__":
    main()
