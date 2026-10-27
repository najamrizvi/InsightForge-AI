# InsightForge AI — Step 21 Error Analysis

## Objective

Investigate where forecasting errors are largest using chronological evaluation windows and shared eligible timestamps. This is a diagnostic experiment, not a model-tuning exercise.

## Data and evaluation

- Source: UCI Individual Household Electric Power Consumption dataset.
- Source attribution: UCI Machine Learning Repository, CC BY 4.0.
- Data processed directly from `electricity_data.zip`; the source archive was not extracted.
- Hourly observations require at least 54 valid minute readings.
- No missing hourly values were imputed.
- Random Forest settings and the 0.80 profile blend weight remained fixed.
- Strict holdout: Windows 2–4, 3,565 shared evaluation hours.
- Window 1 remained outside the strict holdout score.

## Strict holdout results

| Method | MAE (kW) | RMSE (kW) |
|---|---:|---:|
| Previous day | 0.4824 | 0.7294 |
| Previous week | 0.5472 | 0.7937 |
| Hour-of-day profile | 0.5239 | 0.6814 |
| Frozen blend | 0.4873 | 0.6353 |
| Random Forest | 0.4251 | 0.5786 |

Random Forest improved MAE by approximately 11.9% and RMSE by approximately 20.7% relative to the previous-day baseline on this strict holdout.

## Time-of-day findings

- Lowest Random Forest MAE occurred at 05:00: 0.1767 kW.
- Highest Random Forest MAE occurred at 21:00: 0.5923 kW; 19:00 and 20:00 were also difficult.
- Evening errors suggest that some usage patterns are harder to predict from the current features.
- The hourly results pool multiple evaluation windows; time-of-day differences are descriptive and may also reflect seasonal or window-specific effects.
- The Random Forest does not win every individual hour against every baseline.

## Consumption-level findings

Actual-consumption terciles were calculated from strict holdout targets solely for post-prediction error analysis.

| Group | Count | Random Forest MAE | Random Forest RMSE | Blend MAE | Blend RMSE | Previous-day MAE | Previous-day RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low | 1,189 | 0.3549 | 0.4848 | 0.5285 | 0.6525 | 0.2559 | 0.5252 |
| Medium | 1,188 | 0.3718 | 0.4889 | 0.4046 | 0.5086 | 0.4543 | 0.6403 |
| High | 1,188 | 0.5486 | 0.7282 | 0.5288 | 0.7255 | 0.7371 | 0.9543 |

Holdout quantile boundaries:
- Low: actual consumption <= 0.3908 kW.
- Medium: 0.3908 < actual consumption <= 1.0944 kW.
- High: actual consumption > 1.0944 kW.

The high-consumption group has the largest Random Forest errors. The previous-day baseline has lower MAE in the low group, while the frozen blend has slightly lower MAE than Random Forest in the high group. These are descriptive groups only; actual holdout targets were not model features.

## Window robustness

| Window | Shared hours | Previous-day MAE | Frozen-blend MAE | Random Forest MAE |
|---|---:|---:|---:|---:|
| 1 — tuning window, excluded from strict holdout | 1,344 | 0.5779 | 0.4687 | 0.4489 |
| 2 | 1,344 | 0.4154 | 0.5033 | 0.3940 |
| 3 | 966 | 0.3994 | 0.4530 | 0.3573 |
| 4 | 1,255 | 0.6180 | 0.4965 | 0.5105 |

Random Forest has the lowest MAE in Windows 1–3, but not Window 4. Therefore, its aggregate advantage is meaningful but not universal.

## Limitations

- Only one dataset and a limited historical period were evaluated.
- The model and feature set have not been broadly tuned or compared across independent datasets.
- Hourly and consumption-group analyses are descriptive; they do not establish causal explanations.
- Consumption-level boundaries use the evaluation targets after prediction and must not be used as a source of predictive information.
- The findings do not yet establish production readiness or future performance under changed household usage patterns.

## Conclusion

Random Forest is the strongest aggregate candidate among the tested methods on the strict holdout. Its advantage is not universal across evaluation windows, times of day, or consumption levels. Evening and high-consumption periods warrant further investigation before deciding how the eventual application should communicate forecast uncertainty.

Source: UCI Machine Learning Repository, Individual Household Electric Power Consumption, https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
