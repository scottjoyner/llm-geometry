# Gemma + LFM Temporal Drift Observations

## Framework
- Representation drift: 1-CKA, RSA, KNN-overlap, Procrustes residual across shared prompts/layers.
- Weight drift: model fingerprint cosine/L2 over sampled core tensors with deterministic projection.
- PT-vs-IT gap: matched generation/size pairs within Gemma.
- Temporal trajectory: consecutive Gemma-IT checkpoints ordered by generation.

## Pairwise Ranking
| pair_label                                   |   mean_cka |   mean_rsa |   mean_knn |   mean_procrustes |
|:---------------------------------------------|-----------:|-----------:|-----------:|------------------:|
| codegemma_2b vs gemma1_1_2b_it               |   0.985392 |   0.919446 |   0.67451  |        0.00785795 |
| gemma2_2b vs gemma2_2b_it                    |   0.977823 |   0.932575 |   0.792145 |        0.00674563 |
| lfm2_5_1_2b_base vs lfm2_5_1_2b_instruct     |   0.956162 |   0.911581 |   0.812489 |        0.00736414 |
| gemma2_2b vs lfm2_5_1_2b_base                |   0.955926 |   0.746649 |   0.587043 |        0.0125915  |
| gemma2_2b_it vs lfm2_5_1_2b_base             |   0.954213 |   0.777696 |   0.588312 |        0.0125362  |
| gemma2_2b_it vs lfm2_5_1_2b_instruct         |   0.947775 |   0.79654  |   0.58608  |        0.0128635  |
| gemma1_1_2b_it vs gemma2_2b_it               |   0.939065 |   0.798917 |   0.658228 |        0.0118665  |
| lfm2_5_1_2b_base vs lfm2_5_1_2b_thinking     |   0.937944 |   0.833342 |   0.7248   |        0.0099092  |
| gemma1_1_2b_it vs gemma2_2b                  |   0.935516 |   0.761247 |   0.64801  |        0.0123955  |
| gemma2_2b vs lfm2_5_1_2b_instruct            |   0.935481 |   0.753011 |   0.581065 |        0.0135562  |
| codegemma_2b vs gemma2_2b                    |   0.932962 |   0.780767 |   0.63792  |        0.0124827  |
| codegemma_2b vs gemma2_2b_it                 |   0.932958 |   0.796316 |   0.641898 |        0.0122808  |
| gemma1_1_2b_it vs lfm2_5_1_2b_base           |   0.931764 |   0.831132 |   0.581531 |        0.0135003  |
| codegemma_2b vs lfm2_5_1_2b_base             |   0.92835  |   0.817512 |   0.58117  |        0.0135107  |
| lfm2_5_1_2b_instruct vs lfm2_5_1_2b_thinking |   0.91826  |   0.833209 |   0.735506 |        0.0100908  |
| gemma1_1_2b_it vs lfm2_5_1_2b_instruct       |   0.913821 |   0.814273 |   0.582251 |        0.0144707  |
| gemma2_2b vs lfm2_5_1_2b_thinking            |   0.91229  |   0.664223 |   0.569732 |        0.015168   |
| codegemma_2b vs lfm2_5_1_2b_instruct         |   0.910954 |   0.81877  |   0.5763   |        0.0144165  |
| gemma2_2b_it vs lfm2_5_1_2b_thinking         |   0.905107 |   0.686554 |   0.573661 |        0.0152181  |
| codegemma_2b vs lfm2_5_1_2b_thinking         |   0.900916 |   0.779335 |   0.573032 |        0.0153147  |
| gemma1_1_2b_it vs lfm2_5_1_2b_thinking       |   0.898537 |   0.766832 |   0.569759 |        0.0157218  |
| gemma3_1b_pt vs gemma3_4b_pt                 |   0.815519 |   0.834429 |   0.666619 |        0.0237813  |
| gemma2_2b_it vs gemma3_1b_it                 |   0.794141 |   0.797455 |   0.627523 |        0.019907   |
| gemma3_1b_pt vs lfm2_5_1_2b_thinking         |   0.789481 |   0.688919 |   0.514231 |        0.0223443  |
| gemma2_2b vs gemma3_1b_pt                    |   0.778966 |   0.709285 |   0.597003 |        0.0215924  |
| gemma2_2b vs gemma3_1b_it                    |   0.770799 |   0.750915 |   0.610651 |        0.0212537  |
| gemma3_1b_it vs lfm2_5_1_2b_instruct         |   0.76984  |   0.763029 |   0.54325  |        0.0222625  |
| gemma1_1_2b_it vs gemma3_1b_it               |   0.764868 |   0.811239 |   0.588486 |        0.0220859  |
| gemma3_1b_it vs lfm2_5_1_2b_base             |   0.764758 |   0.763984 |   0.541385 |        0.0223579  |
| gemma2_2b_it vs gemma3_1b_pt                 |   0.762704 |   0.739254 |   0.602643 |        0.0221658  |
| gemma1_1_2b_it vs gemma3_1b_pt               |   0.759832 |   0.778472 |   0.569652 |        0.0225612  |
| gemma3_1b_pt vs lfm2_5_1_2b_base             |   0.754539 |   0.734756 |   0.521268 |        0.0227435  |
| codegemma_2b vs gemma3_1b_it                 |   0.749784 |   0.79398  |   0.609962 |        0.0225473  |
| codegemma_2b vs gemma3_1b_pt                 |   0.746439 |   0.760308 |   0.589744 |        0.0224831  |
| gemma3_1b_it vs gemma3_4b_it                 |   0.745211 |   0.844526 |   0.653457 |        0.0272466  |
| gemma2_2b_it vs gemma3_4b_it                 |   0.740194 |   0.7856   |   0.643475 |        0.0253137  |
| gemma2_2b vs gemma3_4b_pt                    |   0.738143 |   0.741334 |   0.61395  |        0.0253872  |
| gemma3_4b_it vs gemma3_4b_pt                 |   0.736055 |   0.955474 |   0.806367 |        0.0206175  |
| gemma3_1b_pt vs lfm2_5_1_2b_instruct         |   0.73506  |   0.720089 |   0.518842 |        0.0228576  |
| gemma3_1b_it vs lfm2_5_1_2b_thinking         |   0.733648 |   0.70593  |   0.536605 |        0.0231676  |
| gemma1_1_2b_it vs gemma3_4b_pt               |   0.729576 |   0.848532 |   0.579058 |        0.0270128  |
| gemma3_4b_pt vs lfm2_5_1_2b_thinking         |   0.72953  |   0.73904  |   0.518754 |        0.0272545  |
| gemma2_2b_it vs gemma3_4b_pt                 |   0.721214 |   0.745559 |   0.623297 |        0.0257035  |
| gemma3_4b_it vs lfm2_5_1_2b_instruct         |   0.715618 |   0.763542 |   0.543554 |        0.0277217  |
| gemma2_2b vs gemma3_4b_it                    |   0.712918 |   0.750287 |   0.621419 |        0.0261315  |
| gemma1_1_2b_it vs gemma3_4b_it               |   0.705979 |   0.857418 |   0.591604 |        0.027699   |
| codegemma_2b vs gemma3_4b_pt                 |   0.703417 |   0.821907 |   0.602294 |        0.0275909  |
| gemma3_4b_pt vs lfm2_5_1_2b_instruct         |   0.696802 |   0.75461  |   0.524684 |        0.0280943  |
| gemma3_1b_it vs gemma3_1b_pt                 |   0.696768 |   0.857431 |   0.650931 |        0.02264    |
| gemma3_4b_it vs lfm2_5_1_2b_base             |   0.694098 |   0.784802 |   0.546273 |        0.0285922  |
| gemma3_4b_pt vs lfm2_5_1_2b_base             |   0.691938 |   0.785368 |   0.52838  |        0.0284427  |
| codegemma_2b vs gemma3_4b_it                 |   0.679953 |   0.812119 |   0.614516 |        0.0283365  |
| gemma3_1b_it vs gemma3_4b_pt                 |   0.674927 |   0.814569 |   0.635032 |        0.0294362  |
| gemma3_4b_it vs lfm2_5_1_2b_thinking         |   0.646993 |   0.717737 |   0.537392 |        0.0296768  |
| gemma3_1b_pt vs gemma3_4b_it                 |   0.624092 |   0.832546 |   0.654915 |        0.0311088  |

## Gemma PT-vs-IT Gap
|   generation |   params_b | model_pt     | model_it     |   rep_gap_1_minus_cka |   rep_mean_cka |   rep_mean_rsa |   weight_cosine |   weight_l2 |
|-------------:|-----------:|:-------------|:-------------|----------------------:|---------------:|---------------:|----------------:|------------:|
|            2 |          2 | gemma2_2b    | gemma2_2b_it |             0.0221775 |       0.977823 |       0.932575 |        0.585509 |     6.75181 |
|            3 |          1 | gemma3_1b_pt | gemma3_1b_it |             0.303232  |       0.696768 |       0.857431 |        0.993582 |    72.8798  |
|            3 |          4 | gemma3_4b_pt | gemma3_4b_it |             0.263945  |       0.736055 |       0.955474 |        0.999989 |     1.22629 |

## Gemma Temporal IT Trajectory
| from_model     | to_model     |   from_generation |   to_generation |   temporal_rep_drift_1_minus_cka |   temporal_rep_rsa |   temporal_weight_cosine |   temporal_weight_l2 |
|:---------------|:-------------|------------------:|----------------:|---------------------------------:|-------------------:|-------------------------:|---------------------:|
| gemma1_1_2b_it | gemma2_2b_it |               1.1 |               2 |                        0.0609348 |           0.798917 |                0.0622496 |              11.0786 |
| gemma2_2b_it   | gemma3_1b_it |               2   |               3 |                        0.205859  |           0.797455 |                0.102274  |             141.4    |
| gemma3_1b_it   | gemma3_4b_it |               3   |               3 |                        0.254789  |           0.844526 |               -0.030349  |             290.695  |

## Family Layer Alignment
| family_a   | family_b   | layer    |   mean_cka |   mean_rsa |   n |
|:-----------|:-----------|:---------|-----------:|-----------:|----:|
| gemma      | gemma      | layer_0  |   0.887487 |   0.892855 |  28 |
| gemma      | gemma      | layer_10 |   0.64993  |   0.871078 |  28 |
| gemma      | gemma      | layer_12 |   0.636982 |   0.866704 |  28 |
| gemma      | gemma      | layer_2  |   0.925315 |   0.72745  |  28 |
| gemma      | gemma      | layer_4  |   0.918819 |   0.687202 |  28 |
| gemma      | gemma      | layer_6  |   0.757933 |   0.792011 |  28 |
| gemma      | gemma      | layer_8  |   0.687338 |   0.820678 |  28 |
| gemma      | lfm        | layer_0  |   0.929501 |   0.889319 |  24 |
| gemma      | lfm        | layer_10 |   0.720111 |   0.765988 |  24 |
| gemma      | lfm        | layer_12 |   0.715747 |   0.770866 |  24 |
| gemma      | lfm        | layer_2  |   0.931866 |   0.713204 |  24 |
| gemma      | lfm        | layer_4  |   0.935325 |   0.640475 |  24 |
| gemma      | lfm        | layer_6  |   0.828637 |   0.771094 |  24 |
| gemma      | lfm        | layer_8  |   0.7189   |   0.749901 |  24 |
| lfm        | lfm        | layer_0  |   0.999408 |   0.995469 |   3 |
| lfm        | lfm        | layer_10 |   0.855964 |   0.791935 |   3 |
| lfm        | lfm        | layer_12 |   0.823286 |   0.675915 |   3 |
| lfm        | lfm        | layer_2  |   0.993981 |   0.970907 |   3 |
| lfm        | lfm        | layer_4  |   0.983166 |   0.860525 |   3 |
| lfm        | lfm        | layer_6  |   0.970295 |   0.867904 |   3 |
| lfm        | lfm        | layer_8  |   0.936088 |   0.852987 |   3 |

## Geometry Health
| model                | family   |   generation | tune_type   |   mean_anisotropy |   mean_participation_ratio |
|:---------------------|:---------|-------------:|:------------|------------------:|---------------------------:|
| gemma1_1_2b_it       | gemma    |          1.1 | it          |          0.221416 |                    8.52242 |
| codegemma_2b         | gemma    |          1.2 | code        |          0.224208 |                    8.19338 |
| gemma2_2b_it         | gemma    |          2   | it          |          0.216377 |                    8.73889 |
| gemma2_2b            | gemma    |          2   | pt          |          0.200516 |                    8.87895 |
| gemma3_1b_it         | gemma    |          3   | it          |          0.388102 |                    5.65637 |
| gemma3_4b_it         | gemma    |          3   | it          |          0.508688 |                    5.23907 |
| gemma3_1b_pt         | gemma    |          3   | pt          |          0.411155 |                    5.39125 |
| gemma3_4b_pt         | gemma    |          3   | pt          |          0.512614 |                    5.05355 |
| lfm2_5_1_2b_base     | lfm      |          2.5 | base        |          0.210488 |                    8.67123 |
| lfm2_5_1_2b_instruct | lfm      |          2.5 | instruct    |          0.249955 |                    7.96572 |
| lfm2_5_1_2b_thinking | lfm      |          2.5 | thinking    |          0.25402  |                    7.66335 |

## Statistical Stability
- Permutation significance rate (p<0.05): 0.6156
- Mean CKA CI width: 0.0394

## Conclusions
- Depth-conditioned drift is required; shallow-layer alignment alone overestimates interchangeability.
- PT-vs-IT gaps can be quantified as a joint representation + weight drift signature rather than a single score.
- LFM2.5 trio serves as a stable in-family baseline for separating instruction-tuning drift from generation drift.