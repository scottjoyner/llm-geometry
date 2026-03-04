# Experiment Plan: Geometric Representation Comparison (Sub-5B Models)

## 1) Scope
Primary target is comparative geometry analysis of hidden-state spaces for compact models, starting with:
- LFM2.5-1.2B-Instruct
- Gemma 3 1B (instruction tuned)
- Qwen2.5 0.5B / 1.5B / 3B

Research hypothesis (inspired by Vec2Vec-style alignment framing): despite architecture/training differences, sufficiently capable models exhibit convergent geometric structure in specific layers and task manifolds.

## 2) Core Questions
- Which layers show strongest cross-model alignment?
- Does alignment increase for semantically constrained prompts (reasoning/math) relative to open-ended prompts?
- How do small-model families differ in isotropy, intrinsic dimensionality, and neighborhood topology?
- Are observed similarities robust across prompt sets and sampling seeds?

## 3) Data and Prompt Design
Prompt packs should be versioned and stratified:
- logical reasoning
- algebraic/mathematical transformations
- analogy/composition
- factual recall
- narrative continuation
- code-like symbolic patterns

Minimum recommended protocol:
- 6 domains x 200 prompts/domain
- fixed random seed for prompt ordering
- max token length constraint uniform across models

## 4) Geometry Extraction Protocol
For each model:
- run deterministic forward pass (no sampling)
- capture hidden states on shared layer index schedule
- compute mean-pooled token embedding per prompt per layer
- save layer matrices as `.npz`

Controls:
- same prompt text across models
- same truncation length
- same precision policy per run group
- fixed tokenizer-side normalization policy

## 5) Metrics
Primary:
- Linear CKA (cross-model representational similarity)
- RSA Spearman on cosine-distance structure
- Participation ratio (intrinsic dimensionality proxy)
- Anisotropy (spectral concentration)

Secondary:
- nearest-neighbor overlap@k
- Procrustes alignment residuals
- geodesic distortion after dimensionality reduction

## 6) Statistical Testing Plan
- bootstrap confidence intervals per metric (1,000+ resamples)
- permutation tests for CKA/RSA deltas between prompt domains
- mixed-effects model:
  - fixed effects: model pair, layer depth, domain
  - random effects: prompt ID

## 7) Stress Tests
- paraphrase invariance test
- tokenization perturbation test
- adversarially irrelevant prefix test
- context length scaling test (short vs long prompt)
- quantization sensitivity test (bf16 vs 8-bit vs 4-bit where feasible)

## 8) Engineering Validation
Script checks to include:
- checksum/sha capture for each downloaded snapshot
- fail-fast on missing layers
- artifact schema validation for `.npz` and CSV outputs
- deterministic rerun test on small prompt subset

## 9) Decision Framework for Utilization
Use geometric comparison outputs to:
- choose candidate model for downstream tasks requiring stable reasoning manifolds
- identify interchangeable model pairs for distillation/ensemble handoff
- detect models with unstable geometry (high anisotropy + low cross-domain consistency)
- prioritize layers for adapter transfer or representation fusion

Operational heuristics:
- high CKA + high RSA in mid/late layers suggests transferable latent structure
- lower anisotropy with stable participation ratio suggests healthier embedding geometry
- strong domain-specific alignment but weak global alignment indicates specialization

## 10) Recommended Next Extensions
- add manifold alignment methods (e.g., orthogonal Procrustes layer mapping)
- add sparse autoencoder features for concept-level geometry comparison
- include cross-lingual prompt packs
- compare base vs instruct checkpoints within each family

## 11) Reproducibility Checklist
- pin package versions
- log hardware, CUDA, and torch versions
- persist model revisions and commit hashes
- store all configs in run metadata
- produce one markdown report per run with conclusions and anomalies

## 12) Minimal Execution Order
1. `python scripts/download_models.py`
2. `python scripts/extract_geometry.py`
3. `python scripts/compare_models.py`
4. review CSV metrics and compute CI/permutation stats in a follow-up notebook/script
