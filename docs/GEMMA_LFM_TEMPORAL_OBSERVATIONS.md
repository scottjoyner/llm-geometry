# Gemma + LFM Temporal Drift Observations

## Framework
- Representation drift: 1-CKA, RSA, KNN-overlap, Procrustes residual across shared prompts/layers.
- Weight drift: model fingerprint cosine/L2 over sampled core tensors with deterministic projection.
- PT-vs-IT gap: matched generation/size pairs within Gemma.
- Temporal trajectory: consecutive Gemma-IT checkpoints ordered by generation.

## Pairwise Ranking
| pair_label                                   |   mean_cka |   mean_rsa |   mean_knn |   mean_procrustes |
|:---------------------------------------------|-----------:|-----------:|-----------:|------------------:|
| codegemma_2b vs gemma1_1_2b_it               |   0.985697 |   0.92764  |   0.715414 |        0.0102064  |
| gemma2_2b vs gemma2_2b_it                    |   0.97757  |   0.937043 |   0.824594 |        0.00863129 |
| gemma2_2b vs lfm2_5_1_2b_base                |   0.959745 |   0.778271 |   0.670547 |        0.0163608  |
| gemma2_2b_it vs lfm2_5_1_2b_base             |   0.956811 |   0.80394  |   0.671056 |        0.0163555  |
| lfm2_5_1_2b_base vs lfm2_5_1_2b_instruct     |   0.954732 |   0.896728 |   0.845909 |        0.00967238 |
| gemma2_2b_it vs lfm2_5_1_2b_instruct         |   0.951852 |   0.822606 |   0.667901 |        0.016598   |
| lfm2_5_1_2b_base vs lfm2_5_1_2b_thinking     |   0.941673 |   0.813724 |   0.770694 |        0.0126834  |
| gemma2_2b vs lfm2_5_1_2b_instruct            |   0.940591 |   0.777215 |   0.662759 |        0.0175531  |
| gemma1_1_2b_it vs gemma2_2b_it               |   0.930727 |   0.794059 |   0.69973  |        0.0168697  |
| gemma1_1_2b_it vs gemma2_2b                  |   0.928184 |   0.753992 |   0.694974 |        0.0175938  |
| gemma1_1_2b_it vs lfm2_5_1_2b_base           |   0.923309 |   0.810665 |   0.658837 |        0.0189521  |
| codegemma_2b vs gemma2_2b_it                 |   0.920034 |   0.787185 |   0.672647 |        0.0178365  |
| codegemma_2b vs gemma2_2b                    |   0.918713 |   0.769284 |   0.660498 |        0.0182606  |
| gemma2_2b vs lfm2_5_1_2b_thinking            |   0.9143   |   0.662758 |   0.648674 |        0.0199826  |
| codegemma_2b vs lfm2_5_1_2b_base             |   0.913005 |   0.799177 |   0.641696 |        0.0193933  |
| lfm2_5_1_2b_instruct vs lfm2_5_1_2b_thinking |   0.912563 |   0.814295 |   0.785236 |        0.0134938  |
| gemma2_2b_it vs lfm2_5_1_2b_thinking         |   0.908378 |   0.69571  |   0.65105  |        0.0199868  |
| gemma1_1_2b_it vs lfm2_5_1_2b_instruct       |   0.903295 |   0.807228 |   0.653809 |        0.0204096  |
| codegemma_2b vs lfm2_5_1_2b_instruct         |   0.896828 |   0.819999 |   0.642083 |        0.0204233  |
| gemma1_1_2b_it vs lfm2_5_1_2b_thinking       |   0.890693 |   0.727365 |   0.647062 |        0.0219973  |
| codegemma_2b vs lfm2_5_1_2b_thinking         |   0.886405 |   0.751623 |   0.638101 |        0.0216427  |
| gemma3_1b_pt vs gemma3_4b_pt                 |   0.86392  |   0.85765  |   0.695554 |        0.0285223  |
| gemma3_1b_it vs gemma3_4b_it                 |   0.809953 |   0.87092  |   0.683363 |        0.0326789  |
| gemma2_2b_it vs gemma3_1b_it                 |   0.801778 |   0.809918 |   0.662033 |        0.0262441  |
| gemma2_2b vs gemma3_1b_pt                    |   0.788246 |   0.730233 |   0.646659 |        0.0289235  |
| gemma2_2b vs gemma3_1b_it                    |   0.776727 |   0.760927 |   0.649534 |        0.0286029  |
| gemma3_1b_it vs lfm2_5_1_2b_instruct         |   0.775059 |   0.776513 |   0.62403  |        0.0297658  |
| gemma3_1b_pt vs lfm2_5_1_2b_thinking         |   0.772761 |   0.691005 |   0.590614 |        0.0307195  |
| gemma2_2b_it vs gemma3_1b_pt                 |   0.764278 |   0.762221 |   0.658227 |        0.0300837  |
| gemma3_1b_it vs lfm2_5_1_2b_base             |   0.761221 |   0.770065 |   0.619041 |        0.0306056  |
| gemma1_1_2b_it vs gemma3_1b_it               |   0.760865 |   0.817689 |   0.636742 |        0.0309632  |
| gemma2_2b_it vs gemma3_4b_it                 |   0.758591 |   0.814744 |   0.700356 |        0.0331555  |
| gemma3_1b_pt vs lfm2_5_1_2b_base             |   0.757309 |   0.74107  |   0.597542 |        0.0309126  |
| gemma2_2b vs gemma3_4b_pt                    |   0.746633 |   0.773329 |   0.674082 |        0.0336747  |
| gemma1_1_2b_it vs gemma3_1b_pt               |   0.745304 |   0.777994 |   0.615784 |        0.0326475  |
| gemma3_4b_pt vs lfm2_5_1_2b_thinking         |   0.745098 |   0.713242 |   0.604795 |        0.0361966  |
| codegemma_2b vs gemma3_1b_it                 |   0.743473 |   0.795175 |   0.61861  |        0.0318084  |
| gemma3_1b_pt vs lfm2_5_1_2b_instruct         |   0.743414 |   0.736421 |   0.593451 |        0.0306054  |
| gemma2_2b vs gemma3_4b_it                    |   0.736164 |   0.778828 |   0.675795 |        0.0341856  |
| gemma3_4b_it vs lfm2_5_1_2b_instruct         |   0.732579 |   0.766354 |   0.629436 |        0.0368842  |
| gemma3_1b_it vs lfm2_5_1_2b_thinking         |   0.729575 |   0.702751 |   0.625029 |        0.0314596  |
| codegemma_2b vs gemma3_1b_pt                 |   0.727671 |   0.751524 |   0.599385 |        0.0326416  |
| gemma2_2b_it vs gemma3_4b_pt                 |   0.72049  |   0.777056 |   0.682502 |        0.0344198  |
| gemma1_1_2b_it vs gemma3_4b_pt               |   0.713147 |   0.843252 |   0.634304 |        0.038307   |
| gemma3_4b_it vs lfm2_5_1_2b_base             |   0.710437 |   0.78948  |   0.629839 |        0.0382165  |
| gemma3_1b_it vs gemma3_4b_pt                 |   0.706425 |   0.838107 |   0.664306 |        0.0376559  |
| gemma1_1_2b_it vs gemma3_4b_it               |   0.705458 |   0.854685 |   0.64441  |        0.0386674  |
| gemma3_4b_pt vs lfm2_5_1_2b_base             |   0.704916 |   0.791577 |   0.612894 |        0.0380217  |
| gemma3_1b_it vs gemma3_1b_pt                 |   0.703581 |   0.872789 |   0.67422  |        0.0303872  |
| gemma3_4b_pt vs lfm2_5_1_2b_instruct         |   0.701631 |   0.749677 |   0.61336  |        0.0377883  |
| gemma3_4b_it vs gemma3_4b_pt                 |   0.68875  |   0.955074 |   0.827009 |        0.0313926  |
| codegemma_2b vs gemma3_4b_it                 |   0.68777  |   0.816746 |   0.631235 |        0.0393116  |
| codegemma_2b vs gemma3_4b_pt                 |   0.687397 |   0.815849 |   0.618513 |        0.0391625  |
| gemma3_1b_pt vs gemma3_4b_it                 |   0.667522 |   0.855819 |   0.692436 |        0.0395023  |
| gemma3_4b_it vs lfm2_5_1_2b_thinking         |   0.660525 |   0.703594 |   0.621902 |        0.0398547  |

## Gemma PT-vs-IT Gap
|   generation |   params_b | model_pt     | model_it     |   rep_gap_1_minus_cka |   rep_mean_cka |   rep_mean_rsa |   weight_cosine |   weight_l2 |
|-------------:|-----------:|:-------------|:-------------|----------------------:|---------------:|---------------:|----------------:|------------:|
|            2 |          2 | gemma2_2b    | gemma2_2b_it |             0.0224303 |       0.97757  |       0.937043 |        0.799968 |     4.89981 |
|            3 |          1 | gemma3_1b_pt | gemma3_1b_it |             0.296419  |       0.703581 |       0.872789 |        0.991361 |    85.9941  |
|            3 |          4 | gemma3_4b_pt | gemma3_4b_it |             0.31125   |       0.68875  |       0.955074 |        0.999991 |     1.10056 |

## Gemma Temporal IT Trajectory
| from_model     | to_model     |   from_generation |   to_generation |   temporal_rep_drift_1_minus_cka |   temporal_rep_rsa |   temporal_weight_cosine |   temporal_weight_l2 |
|:---------------|:-------------|------------------:|----------------:|---------------------------------:|-------------------:|-------------------------:|---------------------:|
| gemma1_1_2b_it | gemma2_2b_it |               1.1 |               2 |                        0.0692731 |           0.794059 |               -0.0935638 |              12.3663 |
| gemma2_2b_it   | gemma3_1b_it |               2   |               3 |                        0.198222  |           0.809918 |               -0.0706334 |             164.803  |
| gemma3_1b_it   | gemma3_4b_it |               3   |               3 |                        0.190047  |           0.87092  |                0.0105048 |             283.333  |

## Family Layer Alignment
| family_a   | family_b   | layer    |   mean_cka |   mean_rsa |   n |
|:-----------|:-----------|:---------|-----------:|-----------:|----:|
| gemma      | gemma      | layer_0  |   0.907104 |   0.885396 |  28 |
| gemma      | gemma      | layer_12 |   0.633259 |   0.873999 |  28 |
| gemma      | gemma      | layer_4  |   0.91924  |   0.689835 |  28 |
| gemma      | gemma      | layer_8  |   0.678264 |   0.822161 |  28 |
| gemma      | lfm        | layer_0  |   0.940159 |   0.880297 |  24 |
| gemma      | lfm        | layer_12 |   0.714624 |   0.764742 |  24 |
| gemma      | lfm        | layer_4  |   0.935547 |   0.637273 |  24 |
| gemma      | lfm        | layer_8  |   0.716292 |   0.749073 |  24 |
| lfm        | lfm        | layer_0  |   0.999417 |   0.995083 |   3 |
| lfm        | lfm        | layer_12 |   0.82703  |   0.663344 |   3 |
| lfm        | lfm        | layer_4  |   0.98302  |   0.854107 |   3 |
| lfm        | lfm        | layer_8  |   0.935824 |   0.853795 |   3 |

## Geometry Health
| model                | family   |   generation | tune_type   |   mean_anisotropy |   mean_participation_ratio |
|:---------------------|:---------|-------------:|:------------|------------------:|---------------------------:|
| gemma1_1_2b_it       | gemma    |          1.1 | it          |          0.219164 |                    8.44716 |
| codegemma_2b         | gemma    |          1.2 | code        |          0.234063 |                    7.90053 |
| gemma2_2b_it         | gemma    |          2   | it          |          0.216614 |                    8.57352 |
| gemma2_2b            | gemma    |          2   | pt          |          0.200928 |                    8.75113 |
| gemma3_1b_it         | gemma    |          3   | it          |          0.400023 |                    5.50198 |
| gemma3_4b_it         | gemma    |          3   | it          |          0.474096 |                    5.70327 |
| gemma3_1b_pt         | gemma    |          3   | pt          |          0.404777 |                    5.61411 |
| gemma3_4b_pt         | gemma    |          3   | pt          |          0.486475 |                    5.35706 |
| lfm2_5_1_2b_base     | lfm      |          2.5 | base        |          0.211092 |                    8.50906 |
| lfm2_5_1_2b_instruct | lfm      |          2.5 | instruct    |          0.251529 |                    7.78053 |
| lfm2_5_1_2b_thinking | lfm      |          2.5 | thinking    |          0.252957 |                    7.50596 |

## Statistical Stability
- Permutation significance rate (p<0.05): 0.5227
- Mean CKA CI width: 0.0414

## Conclusions
- Depth-conditioned drift is required; shallow-layer alignment alone overestimates interchangeability.
- PT-vs-IT gaps can be quantified as a joint representation + weight drift signature rather than a single score.
- LFM2.5 trio serves as a stable in-family baseline for separating instruction-tuning drift from generation drift.