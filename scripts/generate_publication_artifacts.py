from __future__ import annotations

from pathlib import Path

import pandas as pd


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main() -> None:
    out_reports = Path("outputs/reports_1200")
    docs = Path("docs")
    notebooks = Path("notebooks")
    docs.mkdir(parents=True, exist_ok=True)
    notebooks.mkdir(parents=True, exist_ok=True)

    pair = pd.read_csv(out_reports / "pairwise_similarity.csv")
    summary = pd.read_csv(out_reports / "geometry_summary.csv")
    domain = pd.read_csv(out_reports / "domain_similarity.csv")
    ci = pd.read_csv(out_reports / "pairwise_bootstrap_ci.csv")
    perm = pd.read_csv(out_reports / "permutation_tests.csv")

    top_pair = pair.loc[pair["cka"].idxmax()]
    low_pair = pair.loc[pair["cka"].idxmin()]

    mean_pair = (
        pair.groupby(["model_a", "model_b"], as_index=False)
        .agg(mean_cka=("cka", "mean"), mean_rsa=("rsa_spearman", "mean"), mean_knn=("knn_overlap", "mean"))
        .sort_values("mean_cka", ascending=False)
    )

    anis = summary.groupby("model", as_index=False).agg(
        mean_anisotropy=("anisotropy", "mean"),
        mean_pr=("participation_ratio", "mean"),
    ).sort_values("mean_anisotropy")

    sig_frac = float((perm["p_value"] < 0.05).mean())

    ref_domains = {"logical_reasoning", "algebraic_geometry"}
    con_domains = {"narrative_structure", "factual_recall"}
    d_ref = domain[domain["domain"].isin(ref_domains)].groupby(["model_a", "model_b", "layer"])["cka"].mean().rename("ref")
    d_con = domain[domain["domain"].isin(con_domains)].groupby(["model_a", "model_b", "layer"])["cka"].mean().rename("contrast")
    d_join = pd.concat([d_ref, d_con], axis=1).dropna()
    d_join["delta"] = d_join["ref"] - d_join["contrast"]

    cka_ci = ci[ci["metric"] == "cka"].copy()
    cka_ci["width"] = cka_ci["ci_high"] - cka_ci["ci_low"]
    ci_mean_width = float(cka_ci["width"].mean())

    abstract = f"""# Abstract

We evaluate representational geometry across five compact language models (LFM2.5-1.2B-Instruct, Gemma-3-1B-IT, Qwen2.5-0.5B-Instruct, Qwen3-0.6B, Qwen3.5-0.8B) using a deterministic 1,200-prompt benchmark spanning six domains (200 prompts/domain). For each model and layer, we extract pooled hidden-state embeddings and estimate geometric similarity using linear CKA, RSA (Spearman over cosine dissimilarities), neighborhood-overlap topology, and orthogonal Procrustes residuals. We augment point estimates with bootstrap confidence intervals and permutation testing over domain labels. Results show near-universal convergence at early representation layers (max CKA={fmt(float(top_pair['cka']))}) while deeper-layer alignment becomes highly pair-dependent, with the weakest observed alignment at CKA={fmt(float(low_pair['cka']))}. Models with lower anisotropy and higher participation ratio (notably LFM2.5 and Qwen3.5-0.8B) exhibit more stable cross-model geometry than highly concentrated spaces. Domain-conditioned tests indicate non-random structure ({fmt(sig_frac*100)}% of layer-pair tests significant at p<0.05), supporting the claim that alignment depends on reasoning context rather than architecture alone. We introduce a Geometry Atlas visualization that jointly exposes manifold projections, alignment rivers, model fingerprints, and domain tension matrices to communicate multidimensional geometry behavior to users.
"""

    observations = f"""# Full Observations (1200-Prompt Benchmark)

## Experimental Inventory
- Models: {", ".join(sorted(summary['model'].unique()))}
- Layers analyzed: {", ".join(sorted(summary['layer'].unique()))}
- Pairwise rows: {len(pair)}
- Domain-conditional rows: {len(domain)}
- Bootstrap rows: {len(ci)}
- Permutation rows: {len(perm)}

## Global Alignment Findings
- Strongest single alignment: `{top_pair['model_a']} vs {top_pair['model_b']}` at `{top_pair['layer']}` with CKA={fmt(float(top_pair['cka']))}, RSA={fmt(float(top_pair['rsa_spearman']))}, KNN-overlap={fmt(float(top_pair['knn_overlap']))}.
- Weakest single alignment: `{low_pair['model_a']} vs {low_pair['model_b']}` at `{low_pair['layer']}` with CKA={fmt(float(low_pair['cka']))}, RSA={fmt(float(low_pair['rsa_spearman']))}, KNN-overlap={fmt(float(low_pair['knn_overlap']))}.
- Mean CKA ranking by model-pair:
{mean_pair.to_markdown(index=False)}

## Geometry Health (Per-Model)
- Lower anisotropy and higher participation ratio indicate more distributed geometry.
- Per-model geometry summary:
{anis.to_markdown(index=False)}

## Statistical Stability
- Mean CKA CI width: {fmt(ci_mean_width)} (bootstrap).
- Permutation significance rate (p<0.05 for domain-variance CKA): {fmt(sig_frac)}.

## Domain-Specific Contrast (Reasoning/Algebra vs Narrative/Factual)
Top positive deltas (reference minus contrast):
{d_join.sort_values('delta', ascending=False).head(10).reset_index().to_markdown(index=False)}

Top negative deltas (reference minus contrast):
{d_join.sort_values('delta', ascending=True).head(10).reset_index().to_markdown(index=False)}

## Interpretation
- Early-layer geometric convergence is robust across all pairs.
- Mid/deep-layer geometry is selective: some pairs preserve high alignment while others diverge sharply.
- Domain-conditioned significance indicates geometry is task-sensitive, not purely model-size-driven.
- Models with concentrated spectra (high anisotropy) tend to exhibit lower cross-family deep-layer CKA.

## Artifacts
- Geometry tensors: `outputs/geometry_1200/*.npz`
- Reports: `outputs/reports_1200/*.csv`
- Visualization: `outputs/reports_1200/geometry_atlas.html`
"""

    paper_tex = f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}
\\usepackage{{amsmath}}
\\title{{Geometric Alignment of Compact Language Models: A 1,200-Prompt Multidomain Benchmark}}
\\author{{Kipne et al.}}
\\date{{March 2026}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
We evaluate representational geometry across five compact language models (LFM2.5-1.2B-Instruct, Gemma-3-1B-IT, Qwen2.5-0.5B-Instruct, Qwen3-0.6B, Qwen3.5-0.8B) using a deterministic 1,200-prompt benchmark spanning six domains. Hidden-state geometry is compared with linear CKA, RSA, neighborhood overlap, and Procrustes residuals, with bootstrap confidence intervals and permutation tests. We find strong early-layer convergence and pair-specific deep-layer divergence, with significant domain effects in {fmt(sig_frac*100)}\\% of tested layer-pairs. We also present a Geometry Atlas visualization to communicate multidimensional geometric behavior.
\\end{{abstract}}

\\section{{Introduction}}
Compact language models are increasingly deployed in resource-constrained systems, yet geometric comparability of their latent spaces remains under-characterized. Inspired by Vec2Vec-style representational alignment hypotheses, we study whether models under 5B parameters coalesce toward common geometric structures.

\\section{{Benchmark and Methods}}
\\textbf{{Models.}} LFM2.5-1.2B-Instruct, Gemma-3-1B-IT, Qwen2.5-0.5B-Instruct, Qwen3-0.6B, Qwen3.5-0.8B.\\
\\textbf{{Prompts.}} 1,200 total, six domains, 200 prompts/domain.\\
\\textbf{{Layers.}} $\\{{0,4,8,12,16\\}}$ pooled hidden-state representations.\\
\\textbf{{Metrics.}} CKA, RSA, KNN overlap@10, Procrustes residual; bootstrap CIs (40 samples, sample size 350) and permutation tests (20 samples).

\\section{{Results}}
\\subsection{{Key Quantitative Outcomes}}
Strongest alignment: {top_pair['model_a']} vs {top_pair['model_b']} at {top_pair['layer']} (CKA={fmt(float(top_pair['cka']))}).\\
Weakest alignment: {low_pair['model_a']} vs {low_pair['model_b']} at {low_pair['layer']} (CKA={fmt(float(low_pair['cka']))}).\\
Mean CKA CI width: {fmt(ci_mean_width)}.

\\subsection{{Pairwise Mean Alignment}}
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lllrrr}}
\\toprule
Model A & Model B & Metric & Mean CKA & Mean RSA & Mean KNN \\\\
\\midrule
"""

    for _, row in mean_pair.iterrows():
        paper_tex += (
            f"{row['model_a']} & {row['model_b']} & pair-mean & {fmt(float(row['mean_cka']))} & "
            f"{fmt(float(row['mean_rsa']))} & {fmt(float(row['mean_knn']))} \\\\\n"
        )

    paper_tex += """\\bottomrule
\\end{tabular}
\\caption{Mean pairwise alignment across sampled layers.}
\\end{table}

\\subsection{Geometry Health}
\\begin{table}[h]
\\centering
\\begin{tabular}{lrr}
\\toprule
Model & Mean anisotropy & Mean participation ratio \\\\
\\midrule
"""

    for _, row in anis.iterrows():
        paper_tex += f"{row['model']} & {fmt(float(row['mean_anisotropy']))} & {fmt(float(row['mean_pr']))} \\\\\n"

    paper_tex += """\\bottomrule
\\end{tabular}
\\caption{Per-model geometry concentration and intrinsic dimensionality proxies.}
\\end{table}

\\section{Visualization Mechanism: Geometry Atlas}
We provide a four-panel interactive visualization:
(1) 3D manifold projection of per-model embeddings,
(2) layer-wise alignment river (CKA heatmap),
(3) radial model fingerprints over layers,
(4) domain-tension matrix.
The generated artifact is available at \\texttt{outputs/reports_1200/geometry_atlas.html}.

\\section{Discussion}
Findings support partial geometric convergence in compact models, strongest at shallow layers and conditional at deeper layers. Domain significance implies downstream transfer decisions should be domain-conditioned rather than globally averaged.

\\section{Reproducibility}
All scripts, configs, prompts, and outputs are in-repo. Main commands:
\\begin{verbatim}
python scripts/generate_benchmark_prompts.py --config configs/benchmark_1200.yaml
python scripts/extract_geometry.py --config configs/models_1200.yaml --prompts data/prompts_1200.csv --out-dir outputs/geometry_1200
python scripts/evaluate_multidim.py --benchmark-config configs/benchmark_1200_fast.yaml --geometry-dir outputs/geometry_1200 --prompt-csv data/prompts_1200.csv --out-dir outputs/reports_1200 --bootstrap-sample-size 350
python scripts/build_geometry_atlas.py --geometry-dir outputs/geometry_1200 --reports-dir outputs/reports_1200 --prompt-csv data/prompts_1200.csv --out-html outputs/reports_1200/geometry_atlas.html --layer layer_12
\\end{verbatim}

\\bibliographystyle{plain}
\\begin{thebibliography}{9}
\\bibitem{cka}
Kornblith et al. Similarity of Neural Network Representations Revisited. ICML 2019.
\\bibitem{rsa}
Kriegeskorte et al. Representational Similarity Analysis. Frontiers in Systems Neuroscience 2008.
\\bibitem{hf}
Hugging Face model cards for LiquidAI, Google Gemma, and Qwen model families.
\\bibitem{vec2vec}
Vec2Vec-style representation alignment work (insert exact citation metadata before submission).
\\end{thebibliography}

\\end{document}
"""

    notebook_md = f"""# Geometry Benchmark Notebook (1200 Prompts)

## 1. Setup
```python
import pandas as pd
import plotly.express as px
from pathlib import Path

reports = Path('outputs/reports_1200')
pair = pd.read_csv(reports / 'pairwise_similarity.csv')
summary = pd.read_csv(reports / 'geometry_summary.csv')
domain = pd.read_csv(reports / 'domain_similarity.csv')
ci = pd.read_csv(reports / 'pairwise_bootstrap_ci.csv')
perm = pd.read_csv(reports / 'permutation_tests.csv')
```

## 2. Abstract Snapshot
{abstract.replace('# Abstract', '').strip()}

## 3. Pairwise Alignment Overview
```python
pair.groupby(['model_a','model_b'])['cka'].mean().sort_values(ascending=False)
```

## 4. Geometry Health by Model
```python
summary.groupby('model')[['anisotropy','participation_ratio']].mean().sort_values('anisotropy')
```

## 5. Domain Tension
```python
domain.pivot_table(index=['model_a','model_b'], columns='domain', values='cka', aggfunc='mean')
```

## 6. Confidence Intervals
```python
cka_ci = ci[ci.metric=='cka'].copy()
cka_ci['width'] = cka_ci['ci_high'] - cka_ci['ci_low']
cka_ci[['model_a','model_b','layer','boot_mean','ci_low','ci_high','width']].head(20)
```

## 7. Permutation Significance
```python
(perm['p_value'] < 0.05).mean()
```

## 8. Visual Atlas
Open:
- `outputs/reports_1200/geometry_atlas.html`

## 9. Full Observations
{observations.replace('# Full Observations (1200-Prompt Benchmark)', '').strip()}
"""

    (docs / "ABSTRACT_1200.md").write_text(abstract, encoding="utf-8")
    (docs / "OBSERVATIONS_1200.md").write_text(observations, encoding="utf-8")
    (docs / "PAPER_1200.tex").write_text(paper_tex, encoding="utf-8")
    (notebooks / "geometry_benchmark_1200.md").write_text(notebook_md, encoding="utf-8")

    print("Wrote:")
    print(docs / "ABSTRACT_1200.md")
    print(docs / "OBSERVATIONS_1200.md")
    print(docs / "PAPER_1200.tex")
    print(notebooks / "geometry_benchmark_1200.md")


if __name__ == "__main__":
    main()
