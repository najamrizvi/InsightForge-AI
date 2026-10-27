# InsightForge AI — Step 20: Strict Chronological Holdout Validation

Date: 2026-10-15
Status: Completed

## Objective

Evaluate whether the Random Forest forecasting improvement persists beyond Window 1, which was used to select the historical-profile blend weight.

Window 1 is excluded from the strict holdout summary. Windows 2–4 are evaluated on 3,565 shared target hours.

## Holdout results — Windows 2–4

| Method | MAE (kW) | RMSE (kW) |
|---|---:|---:|
| Previous day | 0.4824 | 0.7294 |
| Previous week | 0.5472 | 0.7937 |
| Hour-of-day profile | 0.5239 | 0.6814 |
| Frozen blend (80% profile, 20% previous day) | 0.4873 | 0.6353 |
| Random Forest | 0.4251 | 0.5786 |

Lower values are better.

Relative to the previous-day baseline, Random Forest reduced:
- MAE by approximately 11.9%.
- RMSE by approximately 20.7%.

Relative to the frozen blend, Random Forest reduced:
- MAE by approximately 12.8%.
- RMSE by approximately 8.9%.

## Window-level Random Forest results

| Window | Evaluation hours | MAE (kW) | RMSE (kW) |
|---|---:|---:|---:|
| Window 2 | 1,344 | 0.3940 | 0.5070 |
| Window 3 | 966 | 0.3573 | 0.4843 |
| Window 4 | 1,255 | 0.5105 | 0.7037 |

Random Forest had the lowest MAE in Windows 2 and 3, but not Window 4. The pooled result therefore does not mean it wins in every window.

## Evaluation protocol

- Source: UCI Individual Household Electric Power Consumption dataset.
- Input archive: electricity_data.zip, processed directly without extraction.
- Eligible hourly averages require at least 54 valid readings out of 60.
- Eligible hourly readings: 34,143.
- Random Forest settings were fixed: 50 trees, maximum depth 12, minimum leaf size 10, max_features 0.8, one worker, random_state 42.
- Features: previous-day and previous-week consumption, cyclical hour and weekday, weekend indicator, and month.
- Each window's model was trained only on observations preceding that window.
- All methods were evaluated on the same eligible target timestamps within each window.

This is a chronological walk-forward evaluation. As time advances, observations from earlier windows become available to train later-window models. It is not a single model tested on three completely isolated holdout windows.

## Interpretation and limitations

The results provide encouraging evidence that this fixed Random Forest configuration improves pooled forecasting accuracy on later evaluation windows from this household dataset.

The experiment does not establish production readiness, performance on other households, or guaranteed future improvements. Additional robustness checks and reproducibility review are needed before stronger claims.

## Reproduction files

- Experiment script: step19_random_forest_experiment.py
- Holdout script: step20_strict_holdout.py
- Dataset archive: electricity_data.zip
- Previous checkpoint: STEP18_FORECASTING_RESULTS.md
- Previous report: STEP19_RANDOM_FOREST_RESULTS.md

Source attribution:
UCI Machine Learning Repository, Individual Household Electric Power Consumption
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
License: CC BY 4.0
