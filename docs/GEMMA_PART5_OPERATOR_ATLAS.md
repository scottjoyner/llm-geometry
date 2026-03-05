# Gemma Part 5: Operator Atlas

## Framework
Part 5 estimates layerwise linear drift operators W that map source prompt representations to target prompt representations.
Operators are ridge-regularized and evaluated on held-out prompts (train/test split) to avoid trivial overfit in high dimensions.
The report includes transport quality, operator spectra, layer transfer maps, and prompt-level drift modes.

## Edge Summary
| edge_type    | model_from     | model_to             |   generation_from |   generation_to |   mean_operator_r2 |   mean_cka_gain |   mean_procrustes_gain |   mean_spectral_pr |   mean_spectral_entropy |
|:-------------|:---------------|:---------------------|------------------:|----------------:|-------------------:|----------------:|-----------------------:|-------------------:|------------------------:|
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking |               3   |             2.5 |           0.927758 |       0.358237  |             0.0411033  |           164.548  |                 5.22119 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         |               1.1 |             2   |           0.946965 |       0.0610027 |             0.00578373 |           131.716  |                 5.01909 |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         |               2   |             3   |           0.944497 |       0.141907  |             0.018162   |            93.5109 |                 5.00016 |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         |               3   |             3   |           0.956135 |       0.233457  |             0.0320103  |            51.5188 |                 4.79851 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         |               2   |             3   |           0.969835 |       0.218913  |             0.025746   |            90.0841 |                 4.93487 |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         |               3   |             3   |           0.968048 |       0.181761  |             0.0284945  |            52.3474 |                 4.80827 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         |               2   |             2   |           0.968044 |       0.0200757 |             0.00403026 |           198.135  |                 5.26791 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         |               3   |             3   |           0.941763 |       0.238753  |             0.0237886  |            90.4587 |                 5.02598 |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         |               3   |             3   |           0.974401 |       0.240197  |             0.026031   |           101.936  |                 5.17911 |

## Best Transport Layers
| edge_type   | model_from     | model_to     | layer   |   cka_before |   cka_after |     cka_gain |   procrustes_before |   procrustes_after |   procrustes_gain |   operator_r2_train |   operator_r2 |   operator_fro |   operator_trace_mean |   operator_spectral_entropy |   operator_spectral_pr |
|:------------|:---------------|:-------------|:--------|-------------:|------------:|-------------:|--------------------:|-------------------:|------------------:|--------------------:|--------------:|---------------:|----------------------:|----------------------------:|-----------------------:|
| it_lineage  | gemma1_1_2b_it | gemma2_2b_it | layer_0 |     0.775318 |    0.999996 |  0.224678    |         0.0331827   |         0.00106486 |       0.0321178   |            0.999941 |      0.999732 |        5.30621 |          -3.51878e-05 |                     4.06945 |                56.5262 |
| tune_shift  | gemma2_2b      | gemma2_2b_it | layer_0 |     0.999996 |    0.999993 | -3.33786e-06 |         0.000268674 |         0.00198898 |      -0.0017203   |            0.999858 |      0.998995 |        7.14838 |           0.0234091   |                     4.06347 |                57.0974 |
| tune_shift  | gemma3_4b_pt   | gemma3_4b_it | layer_0 |     0.999788 |    0.999971 |  0.000183046 |         0.00153417  |         0.00249833 |      -0.000964157 |            0.999375 |      0.99882  |        7.22551 |           0.0205038   |                     4.04313 |                55.5564 |
| it_lineage  | gemma2_2b_it   | gemma3_1b_it | layer_0 |     0.986422 |    0.999838 |  0.0134158   |         0.0107088   |         0.00323155 |       0.00747728  |            0.999198 |      0.998333 |        4.26983 |          -0.000146035 |                     4.06149 |                56.1752 |
| pt_lineage  | gemma2_2b      | gemma3_1b_pt | layer_0 |     0.98822  |    0.999799 |  0.0115792   |         0.0105456   |         0.00335679 |       0.00718879  |            0.999228 |      0.998027 |        4.42428 |          -6.23916e-05 |                     4.03906 |                54.8074 |
| tune_shift  | gemma3_4b_pt   | gemma3_4b_it | layer_2 |     0.993972 |    0.999658 |  0.00568604  |         0.00681087  |         0.00237881 |       0.00443207  |            0.99992  |      0.997835 |       16.6389  |           0.0670572   |                     5.39167 |               186.021  |
| it_lineage  | gemma3_1b_it   | gemma3_4b_it | layer_0 |     0.993505 |    0.999837 |  0.00633192  |         0.00813552  |         0.00361746 |       0.00451806  |            0.998276 |      0.997582 |        6.85305 |          -0.000134118 |                     4.01825 |                53.2026 |
| pt_lineage  | gemma3_1b_pt   | gemma3_4b_pt | layer_0 |     0.995185 |    0.999847 |  0.00466228  |         0.00727009  |         0.00377651 |       0.00349358  |            0.998392 |      0.997439 |        6.19862 |           8.80797e-05 |                     4.01773 |                53.2003 |
| tune_shift  | gemma3_1b_pt   | gemma3_1b_it | layer_0 |     0.997549 |    0.999887 |  0.00233757  |         0.00568188  |         0.00419288 |       0.001489    |            0.998296 |      0.997131 |        6.31182 |           0.0134718   |                     4.01922 |                53.6386 |
| pt_lineage  | gemma3_1b_pt   | gemma3_4b_pt | layer_2 |     0.848501 |    0.997482 |  0.148981    |         0.032707    |         0.00563925 |       0.0270677   |            0.999239 |      0.99432  |       24.5713  |           0.000264654 |                     4.93589 |                88.0051 |
| it_lineage  | gemma3_1b_it   | gemma3_4b_it | layer_2 |     0.884953 |    0.996912 |  0.111959    |         0.0312256   |         0.00650015 |       0.0247254   |            0.999135 |      0.992766 |       27.0026  |           0.00217217  |                     4.94428 |                91.3256 |
| tune_shift  | gemma3_4b_pt   | gemma3_4b_it | layer_4 |     0.969482 |    0.997361 |  0.0278792   |         0.0141789   |         0.00600724 |       0.00817167  |            0.999947 |      0.991752 |       22.9081  |           0.079641    |                     5.3861  |               158.812  |
| tune_shift  | gemma3_1b_pt   | gemma3_1b_it | layer_2 |     0.935581 |    0.998264 |  0.0626826   |         0.0236489   |         0.00601536 |       0.0176335   |            0.9989   |      0.991306 |       16.3394  |           0.0142618   |                     5.14334 |               123.593  |
| pt_lineage  | gemma3_1b_pt   | gemma3_4b_pt | layer_4 |     0.833114 |    0.989983 |  0.156869    |         0.0329946   |         0.00880873 |       0.0241858   |            0.99951  |      0.987865 |       31.7774  |          -0.00111506  |                     5.03268 |                86.1599 |
| it_lineage  | gemma2_2b_it   | gemma3_1b_it | layer_2 |     0.901049 |    0.99735  |  0.0963009   |         0.0291216   |         0.00783649 |       0.0212851   |            0.998002 |      0.987737 |       35.4688  |           0.000952681 |                     4.94255 |                97.4687 |
| pt_lineage  | gemma2_2b      | gemma3_1b_pt | layer_2 |     0.8212   |    0.995269 |  0.174069    |         0.0333025   |         0.00849066 |       0.0248118   |            0.997946 |      0.987628 |       49.9544  |           0.00107146  |                     4.83018 |                84.7586 |
| pt_lineage  | gemma2_2b      | gemma3_1b_pt | layer_4 |     0.875051 |    0.995694 |  0.120643    |         0.0304803   |         0.00942191 |       0.0210584   |            0.998097 |      0.984276 |       62.1479  |           0.000707526 |                     4.95504 |                94.5452 |
| pt_lineage  | gemma3_1b_pt   | gemma3_4b_pt | layer_6 |     0.683039 |    0.983643 |  0.300604    |         0.0552956   |         0.0116447  |       0.0436509   |            0.9999   |      0.982594 |       54.7768  |           0.00160036  |                     4.93448 |                41.6718 |
| tune_shift  | gemma2_2b      | gemma2_2b_it | layer_2 |     0.973674 |    0.999594 |  0.0259202   |         0.0111367   |         0.00601193 |       0.00512475  |            0.998293 |      0.982092 |       13.151   |           0.0743172   |                     5.439   |               213.51   |
| tune_shift  | gemma3_4b_pt   | gemma3_4b_it | layer_6 |     0.69377  |    0.981264 |  0.287494    |         0.0432689   |         0.0113769  |       0.031892    |            0.999993 |      0.980973 |       32.2532  |           0.0823731   |                     5.33256 |               101.424  |

## Weakest Transport Layers
| edge_type    | model_from     | model_to             | layer    |   cka_before |   cka_after |   cka_gain |   procrustes_before |   procrustes_after |   procrustes_gain |   operator_r2_train |   operator_r2 |   operator_fro |   operator_trace_mean |   operator_spectral_entropy |   operator_spectral_pr |
|:-------------|:---------------|:---------------------|:---------|-------------:|------------:|-----------:|--------------------:|-------------------:|------------------:|--------------------:|--------------:|---------------:|----------------------:|----------------------------:|-----------------------:|
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_10 |     0.520269 |    0.914269 |  0.394     |           0.0599748 |         0.0259644  |       0.0340104   |            0.999993 |      0.872566 |     35.1923    |           0.0171053   |                     5.25389 |                99.112  |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_12 |     0.764623 |    0.899497 |  0.134874  |           0.042209  |         0.0271238  |       0.0150852   |            0.999523 |      0.882492 |    178.083     |          -0.00617494  |                     5.22027 |                94.4444 |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_10 |     0.738568 |    0.908256 |  0.169688  |           0.0422826 |         0.0260747  |       0.0162079   |            0.998741 |      0.883927 |    139.748     |          -0.00878724  |                     5.26747 |               107.487  |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_12 |     0.551396 |    0.899351 |  0.347955  |           0.0751704 |         0.0304227  |       0.0447478   |            0.999999 |      0.892858 |    105.2       |          -0.00407997  |                     4.74789 |                19.3963 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_12 |     0.486378 |    0.928763 |  0.442385  |           0.0593849 |         0.0241182  |       0.0352667   |            0.999999 |      0.894302 |     34.3565    |           0.0153156   |                     5.26708 |                88.4237 |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_6  |     0.605156 |    0.997643 |  0.392487  |           0.0600102 |         0.0122121  |       0.0477981   |            0.999955 |      0.899229 |      0.0446324 |          -1.66742e-08 |                     5.32272 |               170.966  |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_10 |     0.346146 |    0.995083 |  0.648937  |           0.0773501 |         0.0122718  |       0.0650783   |            1        |      0.903506 |      0.0277433 |          -7.1686e-08  |                     5.51549 |               197.092  |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_8  |     0.409245 |    0.994382 |  0.585137  |           0.0664415 |         0.0123705  |       0.054071    |            0.999997 |      0.905259 |      0.0261956 |           1.66874e-07 |                     5.41278 |               183.027  |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_12 |     0.690237 |    0.912099 |  0.221862  |           0.0659565 |         0.028276   |       0.0376805   |            0.999999 |      0.910278 |     77.2021    |           0.00247781  |                     4.82176 |                21.9986 |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_12 |     0.248035 |    0.99677  |  0.748736  |           0.089211  |         0.0120452  |       0.0771658   |            1        |      0.914202 |      0.0293195 |           3.76334e-07 |                     5.67196 |               213.475  |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_8  |     0.491129 |    0.926383 |  0.435253  |           0.0613333 |         0.0227982  |       0.0385351   |            0.999965 |      0.915229 |     36.8912    |           0.0159122   |                     5.16247 |                73.6311 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_12 |     0.977218 |    0.996113 |  0.0188949 |           0.0171119 |         0.01763    |      -0.000518045 |            0.976057 |      0.916389 |     65.2489    |          -0.00137116  |                     5.32446 |               173.117  |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_8  |     0.689017 |    0.914727 |  0.22571   |           0.0530911 |         0.0261689  |       0.0269222   |            0.999978 |      0.916465 |     72.4473    |          -0.00349943  |                     4.95766 |                37.9315 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_10 |     0.974487 |    0.995884 |  0.0213965 |           0.0164251 |         0.0175006  |      -0.00107549  |            0.977836 |      0.922426 |     53.4091    |          -0.000640009 |                     5.28748 |               164.948  |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_8  |     0.712198 |    0.933082 |  0.220883  |           0.0456618 |         0.022165   |       0.0234967   |            0.99778  |      0.926582 |    110.376     |          -0.000311419 |                     5.20511 |                84.3479 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_12 |     0.645567 |    0.952934 |  0.307367  |           0.0485902 |         0.0195501  |       0.0290401   |            0.999469 |      0.934231 |    207.609     |           0.00561959  |                     5.22973 |               109.637  |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_12 |     0.951356 |    0.997589 |  0.0462335 |           0.0191333 |         0.0111322  |       0.00800113  |            0.999853 |      0.934359 |     20.8579    |           0.0833289   |                     5.46278 |               215.205  |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_8  |     0.968505 |    0.996142 |  0.0276368 |           0.0167681 |         0.0164738  |       0.00029437  |            0.977555 |      0.936837 |     45.5661    |           0.000208469 |                     5.22007 |               150.543  |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_4  |     0.92741  |    0.998556 |  0.0711463 |           0.0301908 |         0.00862862 |       0.0215621   |            0.999523 |      0.939552 |      0.0435603 |           3.50447e-07 |                     5.31373 |               174.698  |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_10 |     0.400788 |    0.945434 |  0.544646  |           0.0812658 |         0.022126   |       0.0591398   |            0.999998 |      0.940674 |     69.8073    |          -0.00145977  |                     4.93381 |                31.7218 |

## Domain Drift (Deepest Layer per Edge)
| edge_type    | model_from     | model_to             | layer    | domain              |   mean_drift_norm |   p90_drift_norm |
|:-------------|:---------------|:---------------------|:---------|:--------------------|------------------:|-----------------:|
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_12 | symbolic_code       |          808.052  |        1189.2    |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_12 | symbolic_code       |          800.198  |        1174.23   |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_12 | narrative_structure |          720.227  |         964.95   |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_12 | analogy_composition |          710.726  |         930.359  |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_12 | analogy_composition |          686.978  |         922.957  |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_12 | algebraic_geometry  |          659.608  |         861.567  |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_12 | logical_reasoning   |          643.577  |         827.392  |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_12 | algebraic_geometry  |          572.834  |         775.755  |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_12 | analogy_composition |          514.301  |         729.748  |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_12 | narrative_structure |          465.408  |         747.409  |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_12 | factual_recall      |          448.943  |         740.309  |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_12 | symbolic_code       |          443.285  |         673.651  |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_12 | narrative_structure |          442.289  |         754.181  |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_12 | narrative_structure |          435.162  |         746.248  |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_12 | symbolic_code       |          430.676  |         691.675  |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_12 | logical_reasoning   |          425.119  |         637.484  |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_12 | logical_reasoning   |          396.897  |         684.339  |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_12 | logical_reasoning   |          389.178  |         689.276  |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_12 | factual_recall      |          387.262  |         629.562  |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_12 | algebraic_geometry  |          301.752  |         498.383  |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_12 | algebraic_geometry  |          298.584  |         495.501  |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_12 | factual_recall      |          289.784  |         450.376  |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_12 | factual_recall      |          276.279  |         422.226  |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_12 | algebraic_geometry  |          271.901  |         317.831  |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_12 | analogy_composition |          269.372  |         419.1    |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_12 | narrative_structure |          259.801  |         330.836  |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_12 | algebraic_geometry  |          257.98   |         315.183  |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_12 | analogy_composition |          231.901  |         265.086  |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_12 | analogy_composition |          226.442  |         307.027  |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_12 | narrative_structure |          222.073  |         284.262  |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_12 | analogy_composition |          211.492  |         266.062  |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_12 | symbolic_code       |          191.658  |         241.313  |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_12 | logical_reasoning   |          188.697  |         237.224  |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_12 | symbolic_code       |          185.426  |         259.357  |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_12 | factual_recall      |          185.256  |         218.852  |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_12 | logical_reasoning   |          180.408  |         218.092  |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_12 | factual_recall      |          179.347  |         251.23   |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_12 | symbolic_code       |          164.506  |         203.005  |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_12 | narrative_structure |          147.374  |         176.139  |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_12 | factual_recall      |          146.334  |         211.091  |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_12 | algebraic_geometry  |          136.008  |         148.519  |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_12 | logical_reasoning   |          135.921  |         174.791  |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_12 | analogy_composition |           31.2299 |          33.2359 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_12 | symbolic_code       |           26.9544 |          28.6892 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_12 | narrative_structure |           25.0871 |          26.7941 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_12 | algebraic_geometry  |           24.463  |          25.6838 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_12 | factual_recall      |           23.3153 |          25.1778 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_12 | logical_reasoning   |           22.9791 |          24.6819 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_12 | narrative_structure |           20.1116 |          21.7456 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_12 | symbolic_code       |           18.959  |          21.5424 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_12 | analogy_composition |           17.6608 |          19.583  |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_12 | logical_reasoning   |           16.4904 |          18.9083 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_12 | factual_recall      |           15.3937 |          16.597  |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_12 | algebraic_geometry  |           14.7769 |          15.701  |

## Drift Mode Energy (Top-3)
| edge_type    | model_from     | model_to             | layer    |   mode_rank |   mode_sv_ratio |
|:-------------|:---------------|:---------------------|:---------|------------:|----------------:|
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_0  |           1 |       0.0861288 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_0  |           2 |       0.0745242 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_0  |           3 |       0.0691422 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_2  |           1 |       0.0582962 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_2  |           2 |       0.0512157 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_2  |           3 |       0.0462851 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_4  |           1 |       0.0578393 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_4  |           2 |       0.0514905 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_4  |           3 |       0.0475355 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_6  |           1 |       0.0599915 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_6  |           2 |       0.0565591 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_6  |           3 |       0.0467856 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_8  |           1 |       0.0599117 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_8  |           2 |       0.0546621 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_8  |           3 |       0.0441948 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_10 |           1 |       0.0626344 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_10 |           2 |       0.0493965 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_10 |           3 |       0.0428003 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_12 |           1 |       0.0630368 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_12 |           2 |       0.0469932 |
| it_lineage   | gemma1_1_2b_it | gemma2_2b_it         | layer_12 |           3 |       0.0413631 |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_0  |           1 |       0.0797324 |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_0  |           2 |       0.076364  |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_0  |           3 |       0.0673753 |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_2  |           1 |       0.0753228 |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_2  |           2 |       0.0645652 |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_2  |           3 |       0.0554522 |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_4  |           1 |       0.0853679 |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_4  |           2 |       0.0672009 |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_4  |           3 |       0.056319  |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_6  |           1 |       0.114145  |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_6  |           2 |       0.062286  |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_6  |           3 |       0.0499285 |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_8  |           1 |       0.121506  |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_8  |           2 |       0.0564923 |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_8  |           3 |       0.045334  |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_10 |           1 |       0.0909597 |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_10 |           2 |       0.0524538 |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_10 |           3 |       0.0465728 |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_12 |           1 |       0.121836  |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_12 |           2 |       0.060162  |
| it_lineage   | gemma2_2b_it   | gemma3_1b_it         | layer_12 |           3 |       0.0418056 |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_0  |           1 |       0.0815559 |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_0  |           2 |       0.0755093 |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_0  |           3 |       0.0650409 |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_2  |           1 |       0.0828029 |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_2  |           2 |       0.066186  |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_2  |           3 |       0.0610038 |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_4  |           1 |       0.0737434 |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_4  |           2 |       0.0639843 |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_4  |           3 |       0.0553733 |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_6  |           1 |       0.205042  |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_6  |           2 |       0.0555873 |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_6  |           3 |       0.0447896 |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_8  |           1 |       0.178599  |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_8  |           2 |       0.053253  |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_8  |           3 |       0.0400476 |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_10 |           1 |       0.205549  |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_10 |           2 |       0.0543238 |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_10 |           3 |       0.0380774 |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_12 |           1 |       0.264201  |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_12 |           2 |       0.0504797 |
| it_lineage   | gemma3_1b_it   | gemma3_4b_it         | layer_12 |           3 |       0.0397421 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_0  |           1 |       0.0794862 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_0  |           2 |       0.0768598 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_0  |           3 |       0.0676372 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_2  |           1 |       0.0936174 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_2  |           2 |       0.0661419 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_2  |           3 |       0.0578084 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_4  |           1 |       0.0823931 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_4  |           2 |       0.0762943 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_4  |           3 |       0.0656226 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_6  |           1 |       0.109952  |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_6  |           2 |       0.0780903 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_6  |           3 |       0.0538808 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_8  |           1 |       0.182582  |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_8  |           2 |       0.0586248 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_8  |           3 |       0.0478843 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_10 |           1 |       0.12732   |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_10 |           2 |       0.0567711 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_10 |           3 |       0.0499605 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_12 |           1 |       0.119322  |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_12 |           2 |       0.0591808 |
| pt_lineage   | gemma2_2b      | gemma3_1b_pt         | layer_12 |           3 |       0.047002  |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_0  |           1 |       0.0819151 |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_0  |           2 |       0.0745737 |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_0  |           3 |       0.0652664 |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_2  |           1 |       0.0884819 |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_2  |           2 |       0.0688171 |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_2  |           3 |       0.0613698 |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_4  |           1 |       0.0874223 |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_4  |           2 |       0.0732802 |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_4  |           3 |       0.0612738 |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_6  |           1 |       0.173365  |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_6  |           2 |       0.0660803 |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_6  |           3 |       0.0524355 |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_8  |           1 |       0.209229  |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_8  |           2 |       0.0497469 |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_8  |           3 |       0.0451816 |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_10 |           1 |       0.224849  |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_10 |           2 |       0.0620388 |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_10 |           3 |       0.0404417 |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_12 |           1 |       0.231796  |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_12 |           2 |       0.0495078 |
| pt_lineage   | gemma3_1b_pt   | gemma3_4b_pt         | layer_12 |           3 |       0.0434128 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_0  |           1 |       0.0768228 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_0  |           2 |       0.0684804 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_0  |           3 |       0.0598814 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_2  |           1 |       0.0944465 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_2  |           2 |       0.0527547 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_2  |           3 |       0.0403756 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_4  |           1 |       0.0799808 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_4  |           2 |       0.0498327 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_4  |           3 |       0.0382985 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_6  |           1 |       0.0741779 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_6  |           2 |       0.0491497 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_6  |           3 |       0.0461571 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_8  |           1 |       0.0740668 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_8  |           2 |       0.0429283 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_8  |           3 |       0.0405458 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_10 |           1 |       0.0622095 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_10 |           2 |       0.0424579 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_10 |           3 |       0.0374085 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_12 |           1 |       0.0643409 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_12 |           2 |       0.0463183 |
| tune_shift   | gemma2_2b      | gemma2_2b_it         | layer_12 |           3 |       0.0351884 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_0  |           1 |       0.0791982 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_0  |           2 |       0.0747646 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_0  |           3 |       0.0646318 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_2  |           1 |       0.0774707 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_2  |           2 |       0.0660157 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_2  |           3 |       0.0571669 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_4  |           1 |       0.0707062 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_4  |           2 |       0.065908  |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_4  |           3 |       0.0574295 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_6  |           1 |       0.0844527 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_6  |           2 |       0.0617763 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_6  |           3 |       0.0559185 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_8  |           1 |       0.140175  |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_8  |           2 |       0.0481089 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_8  |           3 |       0.0473077 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_10 |           1 |       0.124082  |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_10 |           2 |       0.0520247 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_10 |           3 |       0.0428158 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_12 |           1 |       0.101633  |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_12 |           2 |       0.0552641 |
| tune_shift   | gemma3_1b_pt   | gemma3_1b_it         | layer_12 |           3 |       0.047596  |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_0  |           1 |       0.0657091 |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_0  |           2 |       0.0628861 |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_0  |           3 |       0.0587098 |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_2  |           1 |       0.0683711 |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_2  |           2 |       0.0528949 |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_2  |           3 |       0.0480776 |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_4  |           1 |       0.115529  |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_4  |           2 |       0.0580606 |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_4  |           3 |       0.045813  |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_6  |           1 |       0.227251  |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_6  |           2 |       0.0470275 |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_6  |           3 |       0.034082  |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_8  |           1 |       0.194722  |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_8  |           2 |       0.0385927 |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_8  |           3 |       0.0327058 |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_10 |           1 |       0.215717  |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_10 |           2 |       0.0354682 |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_10 |           3 |       0.0291207 |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_12 |           1 |       0.323005  |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_12 |           2 |       0.0283118 |
| tune_shift   | gemma3_4b_pt   | gemma3_4b_it         | layer_12 |           3 |       0.0255906 |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_0  |           1 |       0.0802037 |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_0  |           2 |       0.0742814 |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_0  |           3 |       0.0654176 |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_2  |           1 |       0.080251  |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_2  |           2 |       0.0633748 |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_2  |           3 |       0.0617215 |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_4  |           1 |       0.0680723 |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_4  |           2 |       0.0640012 |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_4  |           3 |       0.0568854 |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_6  |           1 |       0.185148  |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_6  |           2 |       0.0556077 |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_6  |           3 |       0.0457273 |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_8  |           1 |       0.161485  |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_8  |           2 |       0.0523953 |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_8  |           3 |       0.0439317 |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_10 |           1 |       0.203771  |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_10 |           2 |       0.0468686 |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_10 |           3 |       0.0391511 |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_12 |           1 |       0.261952  |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_12 |           2 |       0.0422819 |
| cross_family | gemma3_4b_it   | lfm2_5_1_2b_thinking | layer_12 |           3 |       0.0376862 |

## Conclusions
- Operator fit quality identifies where drift is near-linear versus where non-linear shifts dominate.
- Layer transfer maps show whether representational semantics migrate depth-wise during model evolution.
- Prompt-level drift modes expose domain-specific stress points in the lineage transitions.