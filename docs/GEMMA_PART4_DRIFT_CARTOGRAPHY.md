# Gemma Part 4: Drift Cartography

## Framework
Part 4 maps temporal drift using a composite drift index over representation, neighborhood, structural, and weight-space components.

Composite drift index weights:
- rep_drift (1-CKA): 0.40
- rsa_drift (1-RSA): 0.20
- knn_drift (1-KNN overlap): 0.15
- procrustes residual: 0.15
- weight_drift (1-cosine): 0.10

## Path Summary
| path_label   |   steps |   cumulative_rep_drift |   cumulative_drift_index |   mean_velocity |   max_velocity |
|:-------------|--------:|-----------------------:|-------------------------:|----------------:|---------------:|
| it_path      |       3 |               0.521584 |                  1.69561 |        0.136782 |       0.205859 |
| pt_path      |       2 |               0.405515 |                  1.31888 |        0.221034 |       0.221034 |

## Lineage + Tuning Edges
| edge_type   | model_from     | model_to     |   gen_from |   gen_to |   generation_step |   rep_drift |   rsa_drift |   knn_drift |   procrustes |   domain_cka_std |   weight_drift |   weight_l2 |   drift_index |   drift_velocity |
|:------------|:---------------|:-------------|-----------:|---------:|------------------:|------------:|------------:|------------:|-------------:|-----------------:|---------------:|------------:|--------------:|-----------------:|
| it_lineage  | gemma1_1_2b_it | gemma2_2b_it |        1.1 |        2 |               0.9 |   0.0609348 |   0.201083  |    0.341772 |   0.0118665  |        0.111063  |    0.93775     |    11.0786  |      0.394878 |        0.0677053 |
| it_lineage  | gemma2_2b_it   | gemma3_1b_it |        2   |        3 |               1   |   0.205859  |   0.202545  |    0.372477 |   0.019907   |        0.239103  |    0.897726    |   141.4     |      0.621589 |        0.205859  |
| it_lineage  | gemma3_1b_it   | gemma3_4b_it |        3   |        3 |               0   |   0.254789  |   0.155474  |    0.346543 |   0.0272466  |        0.277762  |    1.03035     |   290.695   |      0.679146 |      nan         |
| pt_lineage  | gemma2_2b      | gemma3_1b_pt |        2   |        3 |               1   |   0.221034  |   0.290715  |    0.402997 |   0.0215924  |        0.211973  |    0.895028    |   211.51    |      0.739472 |        0.221034  |
| pt_lineage  | gemma3_1b_pt   | gemma3_4b_pt |        3   |        3 |               0   |   0.184481  |   0.165571  |    0.333381 |   0.0237813  |        0.248684  |    1.02598     |   331.907   |      0.579412 |      nan         |
| tune_shift  | gemma2_2b      | gemma2_2b_it |        2   |        2 |               0   |   0.0221775 |   0.0674253 |    0.207855 |   0.00674563 |        0.0323229 |    0.414491    |     6.75181 |      0.076219 |      nan         |
| tune_shift  | gemma3_1b_pt   | gemma3_1b_it |        3   |        3 |               0   |   0.303232  |   0.142569  |    0.349069 |   0.02264    |        0.310829  |    0.0064182   |    72.8798  |      0.596165 |      nan         |
| tune_shift  | gemma3_4b_pt   | gemma3_4b_it |        3   |        3 |               0   |   0.263945  |   0.0445259 |    0.193633 |   0.0206175  |        0.185749  |    1.07436e-05 |     1.22629 |      0.361451 |      nan         |

## Instruction Shift Over Time
|   generation | pt_model     | it_model     |   rep_drift |   rsa_drift |   knn_drift |   weight_drift |   drift_index |   rep_drift_delta_vs_prev |   drift_index_delta_vs_prev |
|-------------:|:-------------|:-------------|------------:|------------:|------------:|---------------:|--------------:|--------------------------:|----------------------------:|
|            2 | gemma2_2b    | gemma2_2b_it |   0.0221775 |   0.0674253 |    0.207855 |    0.414491    |      0.076219 |               nan         |                  nan        |
|            3 | gemma3_1b_pt | gemma3_1b_it |   0.303232  |   0.142569  |    0.349069 |    0.0064182   |      0.596165 |                 0.281055  |                    0.519946 |
|            3 | gemma3_4b_pt | gemma3_4b_it |   0.263945  |   0.0445259 |    0.193633 |    1.07436e-05 |      0.361451 |                -0.0392879 |                   -0.234714 |

## Model Drift Profile
| model          |   generation | tune_type   |   params_b |   mean_gemma_rep_drift |   mean_gemma_drift_index |   mean_lfm_rep_drift |   mean_lfm_weight_drift |
|:---------------|-------------:|:------------|-----------:|-----------------------:|-------------------------:|---------------------:|------------------------:|
| gemma1_1_2b_it |          1.1 | it          |          2 |               0.168539 |                 0.549001 |            0.0852926 |                1.04279  |
| codegemma_2b   |          1.2 | code        |          2 |               0.181299 |                 0.570085 |            0.0865931 |                1.04258  |
| gemma2_2b_it   |          2   | it          |          2 |               0.1617   |                 0.536572 |            0.0643017 |                1.07704  |
| gemma2_2b      |          2   | pt          |          2 |               0.164696 |                 0.565216 |            0.065434  |                0.956458 |
| gemma3_1b_pt   |          3   | pt          |          1 |               0.259383 |                 0.712156 |            0.240306  |                1.04743  |
| gemma3_1b_it   |          3   | it          |          1 |               0.257643 |                 0.686504 |            0.243918  |                1.04389  |
| gemma3_4b_pt   |          3   | pt          |          4 |               0.268735 |                 0.68691  |            0.29391   |                0.827833 |
| gemma3_4b_it   |          3   | it          |          4 |               0.293657 |                 0.706383 |            0.314431  |                0.827599 |

## Highest Drift Edges
| edge_type   | model_from     | model_to     |   gen_from |   gen_to |   generation_step |   rep_drift |   rsa_drift |   knn_drift |   procrustes |   domain_cka_std |   weight_drift |   weight_l2 |   drift_index |   drift_velocity |
|:------------|:---------------|:-------------|-----------:|---------:|------------------:|------------:|------------:|------------:|-------------:|-----------------:|---------------:|------------:|--------------:|-----------------:|
| pt_lineage  | gemma2_2b      | gemma3_1b_pt |        2   |        3 |               1   |   0.221034  |   0.290715  |    0.402997 |   0.0215924  |        0.211973  |    0.895028    |   211.51    |      0.739472 |        0.221034  |
| it_lineage  | gemma3_1b_it   | gemma3_4b_it |        3   |        3 |               0   |   0.254789  |   0.155474  |    0.346543 |   0.0272466  |        0.277762  |    1.03035     |   290.695   |      0.679146 |      nan         |
| it_lineage  | gemma2_2b_it   | gemma3_1b_it |        2   |        3 |               1   |   0.205859  |   0.202545  |    0.372477 |   0.019907   |        0.239103  |    0.897726    |   141.4     |      0.621589 |        0.205859  |
| tune_shift  | gemma3_1b_pt   | gemma3_1b_it |        3   |        3 |               0   |   0.303232  |   0.142569  |    0.349069 |   0.02264    |        0.310829  |    0.0064182   |    72.8798  |      0.596165 |      nan         |
| pt_lineage  | gemma3_1b_pt   | gemma3_4b_pt |        3   |        3 |               0   |   0.184481  |   0.165571  |    0.333381 |   0.0237813  |        0.248684  |    1.02598     |   331.907   |      0.579412 |      nan         |
| it_lineage  | gemma1_1_2b_it | gemma2_2b_it |        1.1 |        2 |               0.9 |   0.0609348 |   0.201083  |    0.341772 |   0.0118665  |        0.111063  |    0.93775     |    11.0786  |      0.394878 |        0.0677053 |
| tune_shift  | gemma3_4b_pt   | gemma3_4b_it |        3   |        3 |               0   |   0.263945  |   0.0445259 |    0.193633 |   0.0206175  |        0.185749  |    1.07436e-05 |     1.22629 |      0.361451 |      nan         |
| tune_shift  | gemma2_2b      | gemma2_2b_it |        2   |        2 |               0   |   0.0221775 |   0.0674253 |    0.207855 |   0.00674563 |        0.0323229 |    0.414491    |     6.75181 |      0.076219 |      nan         |

## Lowest Drift Edges
| edge_type   | model_from     | model_to     |   gen_from |   gen_to |   generation_step |   rep_drift |   rsa_drift |   knn_drift |   procrustes |   domain_cka_std |   weight_drift |   weight_l2 |   drift_index |   drift_velocity |
|:------------|:---------------|:-------------|-----------:|---------:|------------------:|------------:|------------:|------------:|-------------:|-----------------:|---------------:|------------:|--------------:|-----------------:|
| tune_shift  | gemma2_2b      | gemma2_2b_it |        2   |        2 |               0   |   0.0221775 |   0.0674253 |    0.207855 |   0.00674563 |        0.0323229 |    0.414491    |     6.75181 |      0.076219 |      nan         |
| tune_shift  | gemma3_4b_pt   | gemma3_4b_it |        3   |        3 |               0   |   0.263945  |   0.0445259 |    0.193633 |   0.0206175  |        0.185749  |    1.07436e-05 |     1.22629 |      0.361451 |      nan         |
| it_lineage  | gemma1_1_2b_it | gemma2_2b_it |        1.1 |        2 |               0.9 |   0.0609348 |   0.201083  |    0.341772 |   0.0118665  |        0.111063  |    0.93775     |    11.0786  |      0.394878 |        0.0677053 |
| pt_lineage  | gemma3_1b_pt   | gemma3_4b_pt |        3   |        3 |               0   |   0.184481  |   0.165571  |    0.333381 |   0.0237813  |        0.248684  |    1.02598     |   331.907   |      0.579412 |      nan         |
| tune_shift  | gemma3_1b_pt   | gemma3_1b_it |        3   |        3 |               0   |   0.303232  |   0.142569  |    0.349069 |   0.02264    |        0.310829  |    0.0064182   |    72.8798  |      0.596165 |      nan         |
| it_lineage  | gemma2_2b_it   | gemma3_1b_it |        2   |        3 |               1   |   0.205859  |   0.202545  |    0.372477 |   0.019907   |        0.239103  |    0.897726    |   141.4     |      0.621589 |        0.205859  |
| it_lineage  | gemma3_1b_it   | gemma3_4b_it |        3   |        3 |               0   |   0.254789  |   0.155474  |    0.346543 |   0.0272466  |        0.277762  |    1.03035     |   290.695   |      0.679146 |      nan         |
| pt_lineage  | gemma2_2b      | gemma3_1b_pt |        2   |        3 |               1   |   0.221034  |   0.290715  |    0.402997 |   0.0215924  |        0.211973  |    0.895028    |   211.51    |      0.739472 |        0.221034  |

## Conclusions
- Gemma drift is dominated by generation transitions and within-generation PT->IT tuning shifts at 3B-era checkpoints.
- Instruction tuning drift is not monotonic across generations; it should be tracked as a generation-specific operator, not a constant offset.
- LFM anchors remain useful as an external baseline for measuring how far Gemma checkpoints move from compact-model geometry priors.