from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Part 4 Gemma drift cartography suite.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--models-config", default="configs/models_gemma_part3.yaml")
    p.add_argument("--analysis-config", default="configs/analysis_gemma_part4.yaml")
    p.add_argument("--reports-input", default="outputs/reports_gemma_part3_360")
    p.add_argument("--reports-output", default="outputs/reports_gemma_part4")
    p.add_argument("--out-md", default="docs/GEMMA_PART4_DRIFT_CARTOGRAPHY.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run(
        [
            args.python,
            "scripts/evaluate_gemma_part4.py",
            "--models-config",
            args.models_config,
            "--analysis-config",
            args.analysis_config,
            "--reports-dir",
            args.reports_input,
            "--out-dir",
            args.reports_output,
            "--out-md",
            args.out_md,
        ]
    )

    run(
        [
            args.python,
            "scripts/build_gemma_part4_dashboard.py",
            "--reports-dir",
            args.reports_output,
            "--out-html",
            f"{args.reports_output}/gemma_part4_dashboard.html",
        ]
    )


if __name__ == "__main__":
    main()
