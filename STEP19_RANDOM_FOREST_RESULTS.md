# InsightForge AI — Step 19: Random Forest Forecasting Results

Date: 2026-10-15
Status: Initial machine-learning feasibility experiment completed

## Objective

Evaluate a fixed RandomForestRegressor against four established forecasting baselines on four chronological evaluation windows.

## Dataset and data handling

- Source: UCI Individual Household Electric Power Consumption dataset.
- Input: electricity_data.zip.
- The source file was processed directly from the ZIP archive without extraction.
- Eligible hourly values require at least 54 valid minute readings out of 60.
- Eligible hourly readings: 34,143.
- Four evaluation windows use the same eligible target timestamps across all methods.
- Total pooled evaluation targets: 4,909 hours.
- No missing consumption readings were imputed.

## Fixed Random Forest configuration

- n_estimators: 50
- max_depth: 12
- min_samples_leaf: 10
- max_features: 0.8
- n_jobs: 1
- random_state: 42

Features: previous-day consumption, previous-week consumption, cyclical hour-of-day features, cyclical day-of-week features, weekend indicator, and month.

For each window, training targets were strictly earlier than the window start. Model settings were not tuned on the evaluation windows.

## Pooled results

| Method | MAE (kW) | RMSE (kW) |
|---|---:|---:|
| Previous day | 0.5085 | 0.7555 |
| Previous week | 0.5518 | 0.7962 |
| Hour-of-day profile | 0.5107 | 0.6682 |
| Frozen blend (80% profile, 20% previous day) | 0.4822 | 0.6324 |
| Random Forest | 0.4316 | 0.5850 |

Relative to the previous-day baseline, Random Forest reduced pooled MAE by approximately 15.1% and RMSE by approximately 22.6%.

## Window-level results

| Window | Evaluation hours | RF MAE | RF RMSE |
|---|---:|---:|---:|
| 1 | 1,344 | 0.4489 | 0.6018 |
| 2 | 1,344 | 0.3940 | 0.5070 |
| 3 | 966 | 0.3573 | 0.4843 |
| 4 | 1,255 | 0.5105 | 0.7037 |

Random Forest achieved the lowest MAE in all four windows. In Window 4, the frozen blend had slightly lower MAE (0.4965) and RMSE (0.6897), and the hour-of-day profile also had slightly lower RMSE (0.6965).

## Feature importance

The leading features across windows were consistently:
1. lag168 (previous-week consumption)
2. lag24 (previous-day consumption)
3. hour_cos
4. month
5. hour_sin

Feature importance indicates predictive usefulness within this fitted model; it does not establish causality.

## Interpretation and limitations

The experiment provides promising feasibility evidence that a modest Random Forest can outperform simple baselines on these pooled historical evaluation windows. It does not establish production readiness or generalization to other households or future periods.

Before making stronger claims, review the implementation, preserve reproducible outputs, and assess robustness. The four windows are chronological evaluations from one household dataset, not independent datasets.

## Reproduction

Script: step19_random_forest_experiment.py
Dataset: electricity_data.zip
Environment: Python 3.12.3, scikit-learn 1.9.1, SciPy 1.18.1, NumPy 2.5.4, pandas 3.0.6

Source attribution:
UCI Machine Learning Repository, Individual Household Electric Power Consumption,
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
License: CC BY 4.0
