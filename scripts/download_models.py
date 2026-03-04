from __future__ import annotations

import argparse
import os

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

from llm_geometry.io_utils import ensure_dir, load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download model binaries from Hugging Face.")
    p.add_argument("--config", default="configs/models.yaml")
    p.add_argument("--out-dir", default="data/models")
    p.add_argument("--models", nargs="*", help="Optional model names from config.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    out_dir = ensure_dir(args.out_dir)
    hf_token = os.getenv("HF_TOKEN")
    api = HfApi(token=hf_token)

    target_names = set(args.models or [])
    failures: list[tuple[str, str]] = []

    for m in cfg["models"]:
        if target_names and m["name"] not in target_names:
            continue

        repo_id = m["repo_id"]
        local_dir = out_dir / m["name"]
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            info = api.model_info(repo_id, revision=m.get("revision", "main"))
            print(f"Downloading {m['name']} ({repo_id}) | sha={info.sha}")
            snapshot_download(
                repo_id=repo_id,
                revision=m.get("revision", "main"),
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                allow_patterns=cfg.get("download", {}).get("allow_patterns"),
                resume_download=True,
                token=hf_token,
            )
        except HfHubHTTPError as e:
            failures.append((m["name"], str(e)))
            print(f"Failed {m['name']} ({repo_id}): {e}")

    if failures:
        print("\nDownload failures:")
        for name, error in failures:
            print(f"- {name}: {error}")


if __name__ == "__main__":
    main()
