# Part 2 Drift and Variance Report

## Scope
This report compares requested Part 2 models across parameter-scale groups (1B, 3-5B, 8B, other) using CKA/RSA/topology metrics over sampled layers.

## Model Inventory
| model                |   params_b | size_bucket   | family   |
|:---------------------|-----------:|:--------------|:---------|
| qwen3_5_0_8b         |        0.8 | 1B            | qwen     |
| gemma3_1b_it         |        1   | 1B            | gemma    |
| lfm2_5_1_2b_instruct |        1.2 | 1B            | lfm      |
| phi4_mini_instruct   |        3.8 | 3-5B          | phi      |
| gemma3_4b_it         |        4   | 3-5B          | gemma    |
| qwen3_5_4b           |        4   | 3-5B          | qwen     |
| lfm2_8b_a1b          |        8   | 8B            | lfm      |
| llama3_8b_instruct   |        8   | 8B            | llama    |

## Pairwise Ranking (Mean Across Layers)
| pair_label                           |   mean_cka |   mean_rsa |   mean_knn |   mean_proc |
|:-------------------------------------|-----------:|-----------:|-----------:|------------:|
| qwen3_5_0_8b vs qwen3_5_4b           |   0.976147 |   0.930853 |   0.686556 |   0.0094972 |
| lfm2_5_1_2b_instruct vs qwen3_5_4b   |   0.958694 |   0.87089  |   0.619109 |   0.0120046 |
| lfm2_5_1_2b_instruct vs qwen3_5_0_8b |   0.9391   |   0.853203 |   0.615204 |   0.0134507 |
| lfm2_8b_a1b vs qwen3_5_4b            |   0.917269 |   0.921268 |   0.634759 |   0.0133423 |
| lfm2_8b_a1b vs qwen3_5_0_8b          |   0.904307 |   0.904562 |   0.596765 |   0.01563   |
| lfm2_5_1_2b_instruct vs lfm2_8b_a1b  |   0.898079 |   0.86423  |   0.676266 |   0.0125419 |
| gemma3_1b_it vs qwen3_5_0_8b         |   0.814079 |   0.799531 |   0.569326 |   0.0140748 |
| gemma3_1b_it vs lfm2_5_1_2b_instruct |   0.80427  |   0.786277 |   0.543091 |   0.0211133 |
| gemma3_1b_it vs qwen3_5_4b           |   0.802808 |   0.816235 |   0.552068 |   0.0204353 |
| gemma3_1b_it vs lfm2_8b_a1b          |   0.753919 |   0.835722 |   0.55624  |   0.0226824 |
| gemma3_1b_it vs gemma3_4b_it         |   0.719527 |   0.872839 |   0.649922 |   0.0284123 |
| gemma3_4b_it vs lfm2_5_1_2b_instruct |   0.689903 |   0.791963 |   0.552867 |   0.0295619 |
| gemma3_4b_it vs qwen3_5_0_8b         |   0.672187 |   0.836911 |   0.568686 |   0.032974  |
| gemma3_4b_it vs qwen3_5_4b           |   0.666094 |   0.84054  |   0.578723 |   0.0281892 |
| gemma3_4b_it vs lfm2_8b_a1b          |   0.652232 |   0.853669 |   0.582149 |   0.0303242 |

## Within-vs-Cross Bucket Alignment
| bucket_relation   | layer    |   mean_cka |    std_cka |   mean_rsa |   mean_knn |   mean_proc |   n |
|:------------------|:---------|-----------:|-----------:|-----------:|-----------:|------------:|----:|
| cross             | layer_0  |   0.989766 | 0.00550308 |   0.94512  |   0.772735 |  0.008469   |  11 |
| cross             | layer_12 |   0.717139 | 0.209611   |   0.872923 |   0.556241 |  0.0281113  |  11 |
| cross             | layer_16 |   0.77807  | 0.253962   |   0.894336 |   0.542024 |  0.0252109  |  11 |
| cross             | layer_4  |   0.937537 | 0.0472439  |   0.764652 |   0.607536 |  0.014189   |  11 |
| cross             | layer_8  |   0.64343  | 0.189456   |   0.84076  |   0.55573  |  0.0273862  |  11 |
| within            | layer_0  |   0.984389 | 0.00303238 |   0.9225   |   0.709693 |  0.00985996 |   4 |
| within            | layer_12 |   0.670769 | 0.177774   |   0.815553 |   0.558697 |  0.0253498  |   4 |
| within            | layer_16 |   0.813745 | 0.238686   |   0.859361 |   0.504643 |  0.0222165  |   4 |
| within            | layer_4  |   0.912176 | 0.0422431  |   0.722051 |   0.557469 |  0.0155662  |   4 |
| within            | layer_8  |   0.648349 | 0.179995   |   0.779971 |   0.55243  |  0.0230425  |   4 |

## Bucket-to-Bucket Matrix (Layer-Conditioned)
| bucket_a   | bucket_b   | layer    |   mean_cka |   mean_rsa |   n |
|:-----------|:-----------|:---------|-----------:|-----------:|----:|
| 1B         | 1B         | layer_0  |   0.98491  |   0.918705 |   3 |
| 1B         | 1B         | layer_12 |   0.741198 |   0.786086 |   3 |
| 1B         | 1B         | layer_16 |   0.932985 |   0.836146 |   3 |
| 1B         | 1B         | layer_4  |   0.904001 |   0.74444  |   3 |
| 1B         | 1B         | layer_8  |   0.699321 |   0.779639 |   3 |
| 1B         | 3-5B       | layer_0  |   0.990859 |   0.952093 |   4 |
| 1B         | 3-5B       | layer_12 |   0.782821 |   0.881287 |   4 |
| 1B         | 3-5B       | layer_16 |   0.823995 |   0.890863 |   4 |
| 1B         | 3-5B       | layer_4  |   0.9239   |   0.789995 |   4 |
| 1B         | 3-5B       | layer_8  |   0.799895 |   0.849283 |   4 |
| 1B         | 8B         | layer_0  |   0.991933 |   0.954606 |   2 |
| 1B         | 8B         | layer_12 |   0.733573 |   0.840008 |   2 |
| 1B         | 8B         | layer_16 |   0.945874 |   0.86018  |   2 |
| 1B         | 8B         | layer_4  |   0.929325 |   0.767082 |   2 |
| 1B         | 8B         | layer_8  |   0.529292 |   0.828003 |   2 |
| 3-5B       | 1B         | layer_0  |   0.985225 |   0.923504 |   2 |
| 3-5B       | 1B         | layer_12 |   0.495086 |   0.828646 |   2 |
| 3-5B       | 1B         | layer_16 |   0.464769 |   0.889716 |   2 |
| 3-5B       | 1B         | layer_4  |   0.928692 |   0.612181 |   2 |
| 3-5B       | 1B         | layer_8  |   0.531454 |   0.818138 |   2 |
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
| 8B         | 1B         | layer_0  |   0.990541 |   0.950446 |   1 |
| 8B         | 1B         | layer_12 |   0.930926 |   0.914586 |   1 |
| 8B         | 1B         | layer_16 |   0.962815 |   0.909778 |   1 |
| 8B         | 1B         | layer_4  |   0.970961 |   0.890135 |   1 |
| 8B         | 1B         | layer_8  |   0.66629  |   0.857866 |   1 |
| 8B         | 3-5B       | layer_0  |   0.989628 |   0.946524 |   1 |
| 8B         | 3-5B       | layer_12 |   0.942407 |   0.918803 |   1 |
| 8B         | 3-5B       | layer_16 |   0.977114 |   0.937349 |   1 |
| 8B         | 3-5B       | layer_4  |   0.986969 |   0.912093 |   1 |
| 8B         | 3-5B       | layer_8  |   0.690228 |   0.891572 |   1 |

## Depth Drift (Shallow-to-Deep CKA Drop)
Higher value means stronger drift from early to late layers.
| pair_label                           |   cka_shallow |   cka_deep |   depth_drift |
|:-------------------------------------|--------------:|-----------:|--------------:|
| gemma3_1b_it vs gemma3_4b_it         |      0.996493 |   0.406451 |     0.590042  |
| gemma3_4b_it vs qwen3_5_0_8b         |      0.983575 |   0.451227 |     0.532349  |
| gemma3_4b_it vs qwen3_5_4b           |      0.982826 |   0.456025 |     0.5268    |
| gemma3_4b_it vs lfm2_5_1_2b_instruct |      0.986874 |   0.478312 |     0.508563  |
| gemma3_4b_it vs lfm2_8b_a1b          |      0.989508 |   0.501578 |     0.487929  |
| gemma3_1b_it vs lfm2_5_1_2b_instruct |      0.98257  |   0.923024 |     0.0595457 |
| gemma3_1b_it vs qwen3_5_0_8b         |      0.983242 |   0.929412 |     0.0538296 |
| gemma3_1b_it vs lfm2_8b_a1b          |      0.986656 |   0.933031 |     0.0536258 |
| lfm2_5_1_2b_instruct vs qwen3_5_0_8b |      0.988919 |   0.946519 |     0.0423995 |
| lfm2_5_1_2b_instruct vs lfm2_8b_a1b  |      0.997209 |   0.958717 |     0.0384924 |
| gemma3_1b_it vs qwen3_5_4b           |      0.980523 |   0.944544 |     0.0359794 |
| lfm2_8b_a1b vs qwen3_5_0_8b          |      0.990541 |   0.962815 |     0.0277257 |

## Domain Variance Hotspots
Higher domain_cka_std means stronger domain sensitivity (potential context-conditioned drift).
| model_a      | model_b              | layer    |   domain_cka_mean |   domain_cka_std | pair_label                           |
|:-------------|:---------------------|:---------|------------------:|-----------------:|:-------------------------------------|
| gemma3_1b_it | gemma3_4b_it         | layer_12 |          0.410388 |        0.168004  | gemma3_1b_it vs gemma3_4b_it         |
| gemma3_1b_it | gemma3_4b_it         | layer_8  |          0.29219  |        0.134081  | gemma3_1b_it vs gemma3_4b_it         |
| gemma3_1b_it | qwen3_5_0_8b         | layer_12 |          0.342292 |        0.0752571 | gemma3_1b_it vs qwen3_5_0_8b         |
| gemma3_1b_it | gemma3_4b_it         | layer_16 |          0.335652 |        0.0747893 | gemma3_1b_it vs gemma3_4b_it         |
| gemma3_1b_it | qwen3_5_4b           | layer_12 |          0.342195 |        0.0736544 | gemma3_1b_it vs qwen3_5_4b           |
| gemma3_1b_it | lfm2_5_1_2b_instruct | layer_12 |          0.341771 |        0.0733267 | gemma3_1b_it vs lfm2_5_1_2b_instruct |
| gemma3_1b_it | lfm2_8b_a1b          | layer_16 |          0.728554 |        0.0677026 | gemma3_1b_it vs lfm2_8b_a1b          |
| gemma3_1b_it | qwen3_5_4b           | layer_16 |          0.732502 |        0.0632258 | gemma3_1b_it vs qwen3_5_4b           |
| gemma3_1b_it | lfm2_5_1_2b_instruct | layer_16 |          0.735192 |        0.059779  | gemma3_1b_it vs lfm2_5_1_2b_instruct |
| gemma3_1b_it | qwen3_5_0_8b         | layer_16 |          0.734146 |        0.0597153 | gemma3_1b_it vs qwen3_5_0_8b         |
| gemma3_1b_it | qwen3_5_0_8b         | layer_8  |          0.369533 |        0.0548882 | gemma3_1b_it vs qwen3_5_0_8b         |
| gemma3_1b_it | lfm2_8b_a1b          | layer_12 |          0.311639 |        0.052025  | gemma3_1b_it vs lfm2_8b_a1b          |

## Geometry Health by Model
| model                |   mean_anisotropy |   mean_participation_ratio |
|:---------------------|------------------:|---------------------------:|
| qwen3_5_4b           |          0.219344 |                    8.47668 |
| lfm2_5_1_2b_instruct |          0.242207 |                    8.11134 |
| lfm2_8b_a1b          |          0.248921 |                    7.72532 |
| qwen3_5_0_8b         |          0.268037 |                    7.19651 |
| gemma3_1b_it         |          0.367918 |                    5.83142 |
| gemma3_4b_it         |          0.537683 |                    5.02082 |

## Statistical Stability
- Permutation significance rate (p<0.05): 0.6133
- Mean CKA CI width: 0.0378
- 90th percentile CKA CI width: 0.0907

## Interpretation
- Compare within-bucket vs cross-bucket CKA by layer to separate size-driven variance from family-driven variance.
- Large depth drift with high domain variance indicates unstable geometric transfer under changing task manifolds.
- High anisotropy with low participation ratio suggests representation concentration that can increase brittle behavior.

## Artifacts
- reports: `outputs\reports_part2_360`
- main visualization: `outputs\reports_part2_360\geometry_atlas_part2_focus.html`