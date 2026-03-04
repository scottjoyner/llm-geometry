# Full Observations (1200-Prompt Benchmark)

## Experimental Inventory
- Models: gemma3_1b_it, lfm2_5_1_2b_instruct, qwen2_5_0_5b_instruct, qwen3_0_6b, qwen3_5_0_8b
- Layers analyzed: layer_0, layer_12, layer_16, layer_4, layer_8
- Pairwise rows: 50
- Domain-conditional rows: 300
- Bootstrap rows: 200
- Permutation rows: 50

## Global Alignment Findings
- Strongest single alignment: `qwen2_5_0_5b_instruct vs qwen3_0_6b` at `layer_0` with CKA=0.9947, RSA=0.9756, KNN-overlap=0.8432.
- Weakest single alignment: `qwen2_5_0_5b_instruct vs qwen3_0_6b` at `layer_4` with CKA=0.0610, RSA=0.6929, KNN-overlap=0.6095.
- Mean CKA ranking by model-pair:
| model_a               | model_b               |   mean_cka |   mean_rsa |   mean_knn |
|:----------------------|:----------------------|-----------:|-----------:|-----------:|
| lfm2_5_1_2b_instruct  | qwen3_5_0_8b          |   0.939802 |   0.85345  |   0.513229 |
| gemma3_1b_it          | qwen3_5_0_8b          |   0.814674 |   0.81     |   0.463963 |
| gemma3_1b_it          | lfm2_5_1_2b_instruct  |   0.800841 |   0.795079 |   0.446299 |
| qwen2_5_0_5b_instruct | qwen3_5_0_8b          |   0.645693 |   0.861296 |   0.595419 |
| gemma3_1b_it          | qwen2_5_0_5b_instruct |   0.635974 |   0.813348 |   0.439625 |
| lfm2_5_1_2b_instruct  | qwen2_5_0_5b_instruct |   0.59794  |   0.807969 |   0.4896   |
| qwen3_0_6b            | qwen3_5_0_8b          |   0.462329 |   0.876383 |   0.610511 |
| lfm2_5_1_2b_instruct  | qwen3_0_6b            |   0.45969  |   0.795542 |   0.495337 |
| gemma3_1b_it          | qwen3_0_6b            |   0.357078 |   0.838503 |   0.443633 |
| qwen2_5_0_5b_instruct | qwen3_0_6b            |   0.31486  |   0.857959 |   0.629382 |

## Geometry Health (Per-Model)
- Lower anisotropy and higher participation ratio indicate more distributed geometry.
- Per-model geometry summary:
| model                 |   mean_anisotropy |   mean_pr |
|:----------------------|------------------:|----------:|
| lfm2_5_1_2b_instruct  |          0.243112 |   8.09722 |
| qwen3_5_0_8b          |          0.267665 |   7.24999 |
| gemma3_1b_it          |          0.37267  |   5.88104 |
| qwen2_5_0_5b_instruct |          0.56804  |   3.90098 |
| qwen3_0_6b            |          0.717458 |   3.19561 |

## Statistical Stability
- Mean CKA CI width: 0.0292 (bootstrap).
- Permutation significance rate (p<0.05 for domain-variance CKA): 0.5800.

## Domain-Specific Contrast (Reasoning/Algebra vs Narrative/Factual)
Top positive deltas (reference minus contrast):
| model_a              | model_b               | layer    |      ref |   contrast |     delta |
|:---------------------|:----------------------|:---------|---------:|-----------:|----------:|
| gemma3_1b_it         | qwen3_5_0_8b          | layer_12 | 0.352007 |  0.252342  | 0.0996657 |
| gemma3_1b_it         | qwen3_5_0_8b          | layer_8  | 0.36592  |  0.275287  | 0.0906333 |
| gemma3_1b_it         | qwen3_5_0_8b          | layer_16 | 0.716615 |  0.636076  | 0.0805393 |
| gemma3_1b_it         | lfm2_5_1_2b_instruct  | layer_12 | 0.347137 |  0.268436  | 0.0787011 |
| gemma3_1b_it         | qwen2_5_0_5b_instruct | layer_8  | 0.129839 |  0.0669665 | 0.0628729 |
| gemma3_1b_it         | qwen3_0_6b            | layer_8  | 0.104257 |  0.0480291 | 0.0562276 |
| gemma3_1b_it         | qwen2_5_0_5b_instruct | layer_16 | 0.280552 |  0.242579  | 0.0379729 |
| lfm2_5_1_2b_instruct | qwen3_5_0_8b          | layer_12 | 0.900885 |  0.863145  | 0.0377402 |
| gemma3_1b_it         | lfm2_5_1_2b_instruct  | layer_16 | 0.701566 |  0.666561  | 0.0350041 |
| gemma3_1b_it         | qwen3_0_6b            | layer_16 | 0.205811 |  0.182977  | 0.0228339 |

Top negative deltas (reference minus contrast):
| model_a               | model_b               | layer    |      ref |   contrast |      delta |
|:----------------------|:----------------------|:---------|---------:|-----------:|-----------:|
| qwen3_0_6b            | qwen3_5_0_8b          | layer_4  | 0.287582 |   0.3803   | -0.0927186 |
| qwen2_5_0_5b_instruct | qwen3_5_0_8b          | layer_4  | 0.324583 |   0.412785 | -0.0882013 |
| qwen2_5_0_5b_instruct | qwen3_5_0_8b          | layer_12 | 0.385836 |   0.457176 | -0.0713399 |
| qwen3_0_6b            | qwen3_5_0_8b          | layer_8  | 0.317985 |   0.389106 | -0.0711211 |
| qwen3_0_6b            | qwen3_5_0_8b          | layer_12 | 0.357191 |   0.427041 | -0.06985   |
| qwen3_0_6b            | qwen3_5_0_8b          | layer_16 | 0.372589 |   0.440056 | -0.0674667 |
| qwen2_5_0_5b_instruct | qwen3_5_0_8b          | layer_8  | 0.377032 |   0.442362 | -0.0653298 |
| lfm2_5_1_2b_instruct  | qwen3_0_6b            | layer_4  | 0.278853 |   0.341334 | -0.0624805 |
| lfm2_5_1_2b_instruct  | qwen2_5_0_5b_instruct | layer_4  | 0.315196 |   0.373844 | -0.0586485 |
| qwen2_5_0_5b_instruct | qwen3_5_0_8b          | layer_16 | 0.462458 |   0.513778 | -0.0513202 |

## Interpretation
- Early-layer geometric convergence is robust across all pairs.
- Mid/deep-layer geometry is selective: some pairs preserve high alignment while others diverge sharply.
- Domain-conditioned significance indicates geometry is task-sensitive, not purely model-size-driven.
- Models with concentrated spectra (high anisotropy) tend to exhibit lower cross-family deep-layer CKA.

## Artifacts
- Geometry tensors: `outputs/geometry_1200/*.npz`
- Reports: `outputs/reports_1200/*.csv`
- Visualization: `outputs/reports_1200/geometry_atlas.html`
