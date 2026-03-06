# Gemma 3 4B vs 27B Comparison

## Setup
- Models: `google/gemma-3-4b-it` vs `google/gemma-3-27b-it`
- Prompt set: 24 prompts across 6 domains
- Layers sampled: 0, 6, 12

## Representation Similarity
Geometry comparison artifacts were not generated (likely due extraction constraints for 27B in this environment).

## Weight Fingerprint Drift
| model_a       | model_b      |   weight_cosine |   weight_l2 |   weight_drift_1_minus_cosine |
|:--------------|:-------------|----------------:|------------:|------------------------------:|
| gemma3_27b_it | gemma3_4b_it |       -0.115733 |     477.702 |                       1.11573 |

## Fingerprint Coverage
| model         |   tensors_used |    global_n |   global_rms |   global_abs_mean |
|:--------------|---------------:|------------:|-------------:|------------------:|
| gemma3_4b_it  |             97 | 1.25746e+09 |     0.352569 |         0.0156393 |
| gemma3_27b_it |            162 | 5.97143e+09 |     0.417657 |         0.0106088 |

## Layerwise Weight-Geometry Summary
| model_a      | model_b       |   mean_weight_geom_cosine |   mean_weight_geom_l2 |   min_weight_geom_cosine |   max_weight_geom_cosine |
|:-------------|:--------------|--------------------------:|----------------------:|-------------------------:|-------------------------:|
| gemma3_4b_it | gemma3_27b_it |                  0.908828 |             0.0600761 |                 0.856647 |                  0.98434 |

## Layerwise Weight-Geometry Mapping
| model_a      | model_b       |   layer_a |   mapped_layer_b |   feature_dim_used |   weight_geom_cosine |   weight_geom_l2 |
|:-------------|:--------------|----------:|-----------------:|-------------------:|---------------------:|-----------------:|
| gemma3_4b_it | gemma3_27b_it |         0 |                0 |                 70 |             0.98434  |        0.0322075 |
| gemma3_4b_it | gemma3_27b_it |         4 |                8 |                 70 |             0.856647 |        0.0783104 |
| gemma3_4b_it | gemma3_27b_it |         8 |               16 |                 70 |             0.899506 |        0.0643277 |
| gemma3_4b_it | gemma3_27b_it |        12 |               20 |                 70 |             0.857708 |        0.0827876 |
| gemma3_4b_it | gemma3_27b_it |        16 |               28 |                 49 |             0.890536 |        0.0722052 |
| gemma3_4b_it | gemma3_27b_it |        20 |               36 |                 49 |             0.893096 |        0.0734155 |
| gemma3_4b_it | gemma3_27b_it |        24 |               44 |                 49 |             0.88892  |        0.0692745 |
| gemma3_4b_it | gemma3_27b_it |        28 |               52 |                 49 |             0.952267 |        0.0347096 |
| gemma3_4b_it | gemma3_27b_it |        32 |               60 |                 49 |             0.95643  |        0.033447  |

## Notes
- If representation artifacts are missing while weight drift is present, the 27B checkpoint likely exceeded runtime/memory for hidden-state extraction.
- Weight drift still gives a robust first-order scale comparison between 4B and 27B checkpoints.