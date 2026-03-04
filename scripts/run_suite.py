from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full geometry comparison pipeline.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--skip-download", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_download:
        run([args.python, "scripts/download_models.py"])

    run([args.python, "scripts/extract_geometry.py"])
    run([args.python, "scripts/compare_models.py"])


if __name__ == "__main__":
    main()
