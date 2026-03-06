from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ExtractionConfig:
    layers: list[int]
    batch_size: int
    max_length: int


def read_prompts(path: str | Path, limit: int | None = None) -> list[str]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        prompts: list[str] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "prompt" not in (reader.fieldnames or []):
                raise ValueError(f"CSV prompt file must contain a 'prompt' column: {path}")
            for row in reader:
                prompt = (row.get("prompt") or "").strip()
                if prompt:
                    prompts.append(prompt)
    else:
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if limit is not None:
        return prompts[:limit]
    return prompts


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def extract_model_geometry(
    repo_id: str,
    model_dir: str | Path,
    prompts: list[str],
    out_path: str | Path,
    cfg: ExtractionConfig,
    torch_dtype: str | None = None,
    low_cpu_mem_usage: bool = True,
    use_device_map: bool = True,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    chosen_dtype = dtype_map.get((torch_dtype or "").lower(), None)
    if chosen_dtype is None:
        chosen_dtype = torch.float16 if use_cuda else torch.float32

    model_kwargs = {
        "torch_dtype": chosen_dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    if use_cuda and use_device_map:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
    if not use_cuda:
        model = model.to("cpu")
    model.eval()

    first_device = next(model.parameters()).device
    layer_vectors: dict[int, list[np.ndarray]] = {layer: [] for layer in cfg.layers}

    for prompt_batch in tqdm(list(batched(prompts, cfg.batch_size)), desc=f"extract:{repo_id}"):
        toks = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
        )
        toks = {k: v.to(first_device) for k, v in toks.items()}

        with torch.no_grad():
            out = model(**toks, output_hidden_states=True)

        hidden_states = out.hidden_states
        for layer in cfg.layers:
            if layer >= len(hidden_states):
                continue
            layer_h = hidden_states[layer]
            mask = toks["attention_mask"].unsqueeze(-1)
            pooled = (layer_h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            layer_vectors[layer].append(pooled.detach().cpu().numpy())

    serializable: dict[str, np.ndarray] = {}
    for layer, values in layer_vectors.items():
        if not values:
            continue
        serializable[f"layer_{layer}"] = np.concatenate(values, axis=0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **serializable)
