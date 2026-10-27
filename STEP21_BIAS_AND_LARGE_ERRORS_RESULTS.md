# InsightForge AI — Step 21.2: Forecast Bias and Large Errors

## Objective

Evaluate signed forecast bias, large absolute-error frequencies, and the contribution of extreme errors on the strict chronological holdout.

## Evaluation protocol

- Strict holdout: Windows 2–4 only.
- Shared eligible observations: 3,565 hourly targets.
- Model settings and blend weight remained fixed.
- Hourly aggregation requires at least 54 valid minute readings.
- No missing hourly values were imputed.
- Errors are forecast minus actual for signed bias.
- Threshold counts use strict greater-than comparisons.
- The worst 5% contains ceil(0.05 × 3,565) = 179 observations.

## Strict holdout accuracy

| Method | MAE (kW) | RMSE (kW) |
|---|---:|---:|
| Previous day | 0.4824 | 0.7294 |
| Previous week | 0.5472 | 0.7937 |
| Hour-of-day profile | 0.5239 | 0.6814 |
| Frozen blend | 0.4873 | 0.6353 |
| Random Forest | 0.4251 | 0.5786 |

Random Forest remains the strongest aggregate candidate on this holdout.

## Signed forecast bias

Bias is mean(forecast - actual); positive values indicate overprediction.

| Method | Bias (kW) | Maximum absolute error (kW) |
|---|---:|---:|
| Previous day | +0.0031 | 5.1757 |
| Previous week | +0.0212 | 3.9307 |
| Hour-of-day profile | +0.1747 | 4.3115 |
| Frozen blend | +0.1404 | 4.3709 |
| Random Forest | +0.0631 | 3.4544 |

Random Forest has a modest positive mean bias. Its average overprediction is smaller than that of the profile and frozen-blend methods, but larger than that of the previous-day and previous-week baselines. Mean bias alone does not describe the spread or severity of individual errors.

## Large absolute-error frequency

| Method | >0.5 kW | >1.0 kW | >1.5 kW |
|---|---:|---:|---:|
| Previous day | 1,290 (36.19%) | 579 (16.24%) | 197 (5.53%) |
| Previous week | 1,497 (41.99%) | 658 (18.46%) | 257 (7.21%) |
| Hour-of-day profile | 1,566 (43.93%) | 488 (13.69%) | 116 (3.25%) |
| Frozen blend | 1,486 (41.68%) | 356 (9.99%) | 69 (1.94%) |
| Random Forest | 1,108 (31.08%) | 288 (8.08%) | 84 (2.36%) |

Random Forest has the lowest frequency of errors above 0.5 kW and 1.0 kW among the five methods. However, the frozen blend has fewer errors above 1.5 kW. Thus, Random Forest is not best at every severity threshold.

These thresholds are descriptive experiment cutoffs, not established operational or safety limits.

## Contribution of the worst 5% of absolute errors

| Method | Worst 5% count | Share of total squared error |
|---|---:|---:|
| Previous day | 179 | 43.54% |
| Previous week | 179 | 40.36% |
| Hour-of-day profile | 179 | 32.22% |
| Frozen blend | 179 | 33.34% |
| Random Forest | 179 | 40.31% |

The worst 5% of Random Forest predictions contribute 40.31% of its total squared error. This confirms that a relatively small subset of observations contributes substantially to RMSE. The share is lower than the previous-day baseline but higher than the profile and frozen-blend methods.

## Interpretation

1. Random Forest has the lowest aggregate MAE and RMSE on the strict holdout.
2. Its mean signed bias is positive but relatively modest.
3. It reduces the frequency of errors above 0.5 and 1.0 kW compared with the previous-day baseline.
4. The frozen blend produces fewer errors above 1.5 kW, showing that method rankings depend on the chosen error criterion.
5. Extreme errors materially affect RMSE for all methods, including Random Forest.

## Limitations

- The analysis covers one household dataset and a limited historical period.
- Thresholds are diagnostic, not validated operational limits.
- Aggregate bias can conceal overprediction in some periods and underprediction in others.
- The worst-5% statistic describes concentration of squared error; it does not explain the cause of those errors.
- The holdout analysis is not a guarantee of future performance or production readiness.

## Conclusion

Random Forest is the strongest aggregate forecasting candidate among the evaluated methods, but positive bias and concentrated extreme errors remain relevant limitations. The frozen blend performs better at the most severe threshold tested. Further diagnostics should investigate when the largest errors occur before any model changes are considered.

Dataset source: UCI Machine Learning Repository, Individual Household Electric Power Consumption, CC BY 4.0.
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
