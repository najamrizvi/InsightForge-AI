# InsightForge AI — Forecasting Feasibility Checkpoint

## Project status
- Phase: Feasibility experimentation
- Completed through: Step 18 — Blended forecasting baseline
- Application development: Not started
- Dataset: electricity_data.zip
- Processing: Stream directly from ZIP; do not extract the dataset
- Hourly eligibility: At least 54 valid minute readings out of 60
- Missing readings: Do not impute missing measurements as zero

## Dataset audit
- Source records: 2,075,259
- Valid Global_active_power readings: 2,049,280
- Rows with missing measurements: 25,979
- Eligible hourly observations: 34,143
- Hours with zero valid readings: 421
- Duplicate timestamps: 0
- Out-of-order timestamps: 0
- Missing minute intervals: 0

## Earlier four-window baseline evaluation
Four 8-week evaluation windows; identical eligible target timestamps for all methods.

Pooled evaluation: 4,909 common hours.

| Method | MAE (kW) | RMSE (kW) |
|---|---:|---:|
| Previous day | 0.5085 | 0.7555 |
| Previous week | 0.5518 | 0.7962 |
| Hour-of-day profile | 0.5107 | 0.6682 |

Interpretation:
- Previous-day baseline has the lowest pooled MAE.
- Hour-of-day profile has the lowest pooled RMSE.
- Performance varies by evaluation window.

## Step 18 — Blended forecasting baseline
Blend formula:
prediction = (1 - w) * previous_day + w * historical_hour_of_day_profile

- Candidate weights: 0.00 to 1.00, increments of 0.05
- Weight-selection window: Window 1 only
- Selected profile weight: 0.80
- Frozen weight evaluated on: Windows 2–4
- Corrected window boundaries:
  1. 2010-04-16 22:00 to 2010-06-11 21:00
  2. 2010-06-11 22:00 to 2010-08-06 21:00
  3. 2010-08-06 22:00 to 2010-10-01 21:00
  4. 2010-10-01 22:00 to 2010-11-26 21:00
- Shared target counts: 1,344; 1,344; 966; 1,255
- Holdout target count, Windows 2–4: 3,565

### Frozen-weight holdout results

| Method | MAE (kW) | RMSE (kW) |
|---|---:|---:|
| Previous day | 0.4824 | 0.7294 |
| Previous week | 0.5472 | 0.7937 |
| Hour-of-day profile | 0.5239 | 0.6814 |
| Frozen blend (80% profile) | 0.4873 | 0.6353 |

Compared with previous day:
- MAE improvement: -1.03% (slightly worse)
- RMSE improvement: +12.90%

### Step 18 conclusion
Keep the frozen blend as a candidate, not as a universal replacement.
The blend reduces pooled holdout RMSE but has slightly higher MAE than the previous-day baseline. Window-level results vary. The blend weight was selected on Window 1 only and frozen for Windows 2–4.

The final 21:00 hour on 2010-11-26 has only 3 valid readings and is excluded under the 54-reading rule. The evaluation window still ends at 21:00; its last eligible target can be earlier.

## Next objective
Design a lightweight machine-learning forecasting feasibility experiment.
- Use only information available at prediction time.
- Avoid temporal leakage.
- Compare against existing baselines on identical target timestamps.
- Preserve the same hourly eligibility rule.
- Keep processing memory-conscious.
- Do not start application development until feasibility evidence supports it.

Dataset attribution:
UCI Machine Learning Repository, Individual Household Electric Power Consumption.
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
