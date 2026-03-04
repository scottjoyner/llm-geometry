# LLM Geometry Comparison Toolkit

End-to-end toolkit for representation-geometry analysis of compact LLMs under 5B parameters.

## Model Set
- `LiquidAI/LFM2.5-1.2B-Instruct`
- `google/gemma-3-1b-it`
- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3.5-0.8B`

## Quick Start
```bash
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
set PYTHONPATH=src
```

## 1200-Prompt Benchmark
```bash
python scripts/generate_benchmark_prompts.py --config configs/benchmark_1200.yaml
python scripts/extract_geometry.py --config configs/models_1200.yaml --prompts data/prompts_1200.csv --out-dir outputs/geometry_1200
python scripts/evaluate_multidim.py --benchmark-config configs/benchmark_1200_fast.yaml --geometry-dir outputs/geometry_1200 --prompt-csv data/prompts_1200.csv --out-dir outputs/reports_1200 --bootstrap-sample-size 350
python scripts/build_geometry_atlas.py --geometry-dir outputs/geometry_1200 --reports-dir outputs/reports_1200 --prompt-csv data/prompts_1200.csv --out-html outputs/reports_1200/geometry_atlas.html --layer layer_12
```

## Reader Artifacts
- Root PDF paper: `LLM_Geometry_Benchmark_1200.pdf`
- LaTeX source: `docs/PAPER_1200.tex`
- Abstract: `docs/ABSTRACT_1200.md`
- Full observations: `docs/OBSERVATIONS_1200.md`
- Interactive visualization: `outputs/reports_1200/geometry_atlas.html`

## Notes
- Downloaded model binaries are kept out of git (`data/models/`).
- The repository tracks report artifacts and paper assets for reproducibility.
