from __future__ import annotations

import argparse
from pathlib import Path

from llm_geometry.extraction import ExtractionConfig, extract_model_geometry, read_prompts
from llm_geometry.io_utils import ensure_dir, load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract hidden-state geometric representations.")
    p.add_argument("--config", default="configs/models.yaml")
    p.add_argument("--models-dir", default="data/models")
    p.add_argument("--prompts", default="data/prompts.txt")
    p.add_argument("--out-dir", default="outputs/geometry")
    p.add_argument("--models", nargs="*", help="Optional model names from config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    ext_cfg = cfg["extraction"]

    prompts = read_prompts(args.prompts, limit=ext_cfg.get("max_prompts"))
    out_dir = ensure_dir(args.out_dir)

    selection = set(args.models or [])
    ex_cfg = ExtractionConfig(
        layers=list(ext_cfg.get("layers", [])),
        batch_size=int(ext_cfg.get("batch_size", 4)),
        max_length=int(ext_cfg.get("max_length", 256)),
    )

    failures: list[tuple[str, str]] = []
    for model in cfg["models"]:
        if selection and model["name"] not in selection:
            continue
        if model.get("skip_extraction", False):
            print(f"Skip {model['name']}: skip_extraction=true")
            continue

        model_path = Path(args.models_dir) / model["name"]
        if not model_path.exists():
            print(f"Skip {model['name']}: not downloaded at {model_path}")
            continue

        out_path = out_dir / f"{model['name']}.npz"
        try:
            extract_model_geometry(model["repo_id"], model_path, prompts, out_path, ex_cfg)
            print(f"Saved geometry tensors: {out_path}")
        except Exception as e:
            failures.append((model["name"], str(e)))
            print(f"Failed {model['name']}: {e}")

    if failures:
        print("\nExtraction failures:")
        for name, err in failures:
            print(f"- {name}: {err}")


if __name__ == "__main__":
    main()
