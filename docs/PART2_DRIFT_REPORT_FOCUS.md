# Part 2 Drift and Variance Report

## Scope
This report compares requested Part 2 models across parameter-scale groups (1B, 3-5B, 8B, other) using CKA/RSA/topology metrics over sampled layers.

## Model Inventory
| model        |   params_b | size_bucket   | family   |
|:-------------|-----------:|:--------------|:---------|
| gemma3_4b_it |          4 | 3-5B          | gemma    |
| qwen3_5_4b   |          4 | 3-5B          | qwen     |
| lfm2_8b_a1b  |          8 | 8B            | lfm      |

## Pairwise Ranking (Mean Across Layers)
| pair_label                  |   mean_cka |   mean_rsa |   mean_knn |   mean_proc |
|:----------------------------|-----------:|-----------:|-----------:|------------:|
| lfm2_8b_a1b vs qwen3_5_4b   |   0.917269 |   0.921268 |   0.634759 |   0.0133423 |
| gemma3_4b_it vs qwen3_5_4b  |   0.666094 |   0.84054  |   0.578723 |   0.0281892 |
| gemma3_4b_it vs lfm2_8b_a1b |   0.652232 |   0.853669 |   0.582149 |   0.0303242 |

## Within-vs-Cross Bucket Alignment
| bucket_relation   | layer    |   mean_cka |       std_cka |   mean_rsa |   mean_knn |   mean_proc |   n |
|:------------------|:---------|-----------:|--------------:|-----------:|-----------:|------------:|----:|
| cross             | layer_0  |   0.989568 |   8.48838e-05 |   0.940641 |   0.745594 |  0.0101375  |   2 |
| cross             | layer_12 |   0.684503 |   0.364732    |   0.912556 |   0.570302 |  0.0299948  |   2 |
| cross             | layer_16 |   0.739346 |   0.336254    |   0.932339 |   0.56139  |  0.0263593  |   2 |
| cross             | layer_4  |   0.965154 |   0.0308518   |   0.801267 |   0.599681 |  0.0126114  |   2 |
| cross             | layer_8  |   0.545183 |   0.205125    |   0.85054  |   0.565304 |  0.0300631  |   2 |
| within            | layer_0  |   0.982826 | nan           |   0.933885 |   0.700723 |  0.00780641 |   1 |
| within            | layer_12 |   0.459483 | nan           |   0.903955 |   0.580499 |  0.0437862  |   1 |
| within            | layer_16 |   0.456025 | nan           |   0.929007 |   0.565178 |  0.0395906  |   1 |
| within            | layer_4  |   0.936702 | nan           |   0.654882 |   0.5056   |  0.0163359  |   1 |
| within            | layer_8  |   0.495432 | nan           |   0.780969 |   0.541615 |  0.0334271  |   1 |

## Bucket-to-Bucket Matrix (Layer-Conditioned)
| bucket_a   | bucket_b   | layer    |   mean_cka |   mean_rsa |   n |
|:-----------|:-----------|:---------|-----------:|-----------:|----:|
| 3-5B       | 3-5B       | layer_0  |   0.982826 |   0.933885 |   1 |
| 3-5B       | 3-5B       | layer_12 |   0.459483 |   0.903955 |   1 |
| 3-5B       | 3-5B       | layer_16 |   0.456025 |   0.929007 |   1 |
| 3-5B       | 3-5B       | layer_4  |   0.936702 |   0.654882 |   1 |
| 3-5B       | 3-5B       | layer_8  |   0.495432 |   0.780969 |   1 |
| 3-5B       | 8B         | layer_0  |   0.989508 |   0.934758 |   1 |
| 3-5B       | 8B         | layer_12 |   0.426598 |   0.90631  |   1 |
| 3-5B       | 8B         | layer_16 |   0.501578 |   0.92733  |   1 |
| 3-5B       | 8B         | layer_4  |   0.943338 |   0.690441 |   1 |
| 3-5B       | 8B         | layer_8  |   0.400137 |   0.809507 |   1 |
| 8B         | 3-5B       | layer_0  |   0.989628 |   0.946524 |   1 |
| 8B         | 3-5B       | layer_12 |   0.942407 |   0.918803 |   1 |
| 8B         | 3-5B       | layer_16 |   0.977114 |   0.937349 |   1 |
| 8B         | 3-5B       | layer_4  |   0.986969 |   0.912093 |   1 |
| 8B         | 3-5B       | layer_8  |   0.690228 |   0.891572 |   1 |

## Depth Drift (Shallow-to-Deep CKA Drop)
Higher value means stronger drift from early to late layers.
| pair_label                  |   cka_shallow |   cka_deep |   depth_drift |
|:----------------------------|--------------:|-----------:|--------------:|
| gemma3_4b_it vs qwen3_5_4b  |      0.982826 |   0.456025 |     0.5268    |
| gemma3_4b_it vs lfm2_8b_a1b |      0.989508 |   0.501578 |     0.487929  |
| lfm2_8b_a1b vs qwen3_5_4b   |      0.989628 |   0.977114 |     0.0125138 |

## Domain Variance Hotspots
Higher domain_cka_std means stronger domain sensitivity (potential context-conditioned drift).
| model_a      | model_b     | layer    |   domain_cka_mean |   domain_cka_std | pair_label                  |
|:-------------|:------------|:---------|------------------:|-----------------:|:----------------------------|
| gemma3_4b_it | lfm2_8b_a1b | layer_8  |          0.194963 |        0.0286816 | gemma3_4b_it vs lfm2_8b_a1b |
| lfm2_8b_a1b  | qwen3_5_4b  | layer_8  |          0.544707 |        0.0286518 | lfm2_8b_a1b vs qwen3_5_4b   |
| gemma3_4b_it | qwen3_5_4b  | layer_8  |          0.353078 |        0.0264124 | gemma3_4b_it vs qwen3_5_4b  |
| lfm2_8b_a1b  | qwen3_5_4b  | layer_12 |          0.818172 |        0.0251661 | lfm2_8b_a1b vs qwen3_5_4b   |
| gemma3_4b_it | lfm2_8b_a1b | layer_4  |          0.827136 |        0.0251592 | gemma3_4b_it vs lfm2_8b_a1b |
| gemma3_4b_it | lfm2_8b_a1b | layer_16 |          0.241881 |        0.0248569 | gemma3_4b_it vs lfm2_8b_a1b |
| gemma3_4b_it | qwen3_5_4b  | layer_12 |          0.231257 |        0.0247287 | gemma3_4b_it vs qwen3_5_4b  |
| gemma3_4b_it | qwen3_5_4b  | layer_16 |          0.253666 |        0.0218067 | gemma3_4b_it vs qwen3_5_4b  |
| gemma3_4b_it | qwen3_5_4b  | layer_0  |          0.940138 |        0.0150309 | gemma3_4b_it vs qwen3_5_4b  |
| gemma3_4b_it | lfm2_8b_a1b | layer_12 |          0.185402 |        0.0148491 | gemma3_4b_it vs lfm2_8b_a1b |
| gemma3_4b_it | qwen3_5_4b  | layer_4  |          0.789826 |        0.0133599 | gemma3_4b_it vs qwen3_5_4b  |
| lfm2_8b_a1b  | qwen3_5_4b  | layer_16 |          0.882221 |        0.0111245 | lfm2_8b_a1b vs qwen3_5_4b   |

## Geometry Health by Model
| model        |   mean_anisotropy |   mean_participation_ratio |
|:-------------|------------------:|---------------------------:|
| qwen3_5_4b   |          0.219344 |                    8.47668 |
| lfm2_8b_a1b  |          0.248921 |                    7.72532 |
| gemma3_4b_it |          0.537683 |                    5.02082 |

## Statistical Stability
- Permutation significance rate (p<0.05): 0.5333
- Mean CKA CI width: 0.0455
- 90th percentile CKA CI width: 0.0961

## Interpretation
- Compare within-bucket vs cross-bucket CKA by layer to separate size-driven variance from family-driven variance.
- Large depth drift with high domain variance indicates unstable geometric transfer under changing task manifolds.
- High anisotropy with low participation ratio suggests representation concentration that can increase brittle behavior.

## Artifacts
- reports: `outputs\reports_part2_focus_360`
- main visualization: `outputs\reports_part2_focus_360\geometry_atlas_part2_focus.html`