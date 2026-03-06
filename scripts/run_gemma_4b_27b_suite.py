from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Gemma 3 4B vs 27B comparison suite.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--models-config", default="configs/models_gemma_4b_27b.yaml")
    p.add_argument("--benchmark-config", default="configs/benchmark_gemma_4b_27b_24.yaml")
    p.add_argument("--prompts-csv", default="data/prompts_gemma_4b_27b_24.csv")
    p.add_argument("--models-dir", default="data/models")
    p.add_argument("--geometry-dir", default="outputs/geometry_gemma_4b_27b_24")
    p.add_argument("--reports-dir", default="outputs/reports_gemma_4b_27b_24")
    p.add_argument("--out-md", default="docs/GEMMA_4B_27B_COMPARISON.md")
    p.add_argument("--skip-download", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run([args.python, "scripts/generate_benchmark_prompts.py", "--config", args.benchmark_config, "--out-csv", args.prompts_csv])

    if not args.skip_download:
        run([args.python, "scripts/download_models.py", "--config", args.models_config, "--out-dir", args.models_dir])

    # Extraction can fail for 27B on low-memory hosts; keep flow alive.
    try:
        run(
            [
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
            ]
        )
    except subprocess.CalledProcessError:
        print("Geometry extraction failed for at least one model; continuing with available artifacts.")

    geom_files = list(Path(args.geometry_dir).glob("*.npz"))
    if len(geom_files) >= 2:
        run(
            [
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
                "90",
            ]
        )
    else:
        print("Skipped evaluate_multidim: need at least two geometry npz files.")

    run(
        [
            args.python,
            "scripts/compute_weight_fingerprints.py",
            "--models-config",
            args.models_config,
            "--models-dir",
            args.models_dir,
            "--out-dir",
            args.reports_dir,
            "--layer-stride",
            "6",
            "--max-tensors",
            "180",
        ]
    )

    run(
        [
            args.python,
            "scripts/evaluate_gemma_4b_27b_weight_geometry.py",
            "--models-dir",
            args.models_dir,
            "--model-a",
            "gemma3_4b_it",
            "--model-b",
            "gemma3_27b_it",
            "--out-dir",
            args.reports_dir,
            "--layer-stride",
            "4",
            "--max-tensors-per-layer",
            "24",
        ]
    )

    run([args.python, "scripts/evaluate_gemma_4b_27b.py", "--reports-dir", args.reports_dir, "--out-md", args.out_md])


if __name__ == "__main__":
    main()
