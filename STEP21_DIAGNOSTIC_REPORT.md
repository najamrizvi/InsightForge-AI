# InsightForge AI — Forecasting Error Diagnostic Report

**Phase:** Step 21 — Forecasting Diagnostics  
**Report date:** 2026-10-16  
**Dataset:** Household Electric Power Consumption  
**Target:** Hourly Global Active Power (kW)  
**Model under investigation:** Frozen Random Forest baseline  
**Status:** Diagnostics consolidated; no model changes authorized

---

## 1. Executive Summary

The forecasting diagnostics investigated large prediction errors, temporal
consumption patterns, upward consumption events, and the relationship
between event characteristics and Random Forest forecast errors.

The strict holdout covers Windows 2–4 and contains 3,565 shared eligible
evaluation hours. The Random Forest baseline achieved MAE of 0.4251 kW
and RMSE of 0.5786 kW in the previously recorded holdout evaluation.

The diagnostics identify three areas for further investigation:

1. The model underforecasts upward consumption events.
2. Larger upward changes tend to accompany larger absolute errors overall,
   although this relationship varies substantially by evaluation window.
3. Upward events beginning at higher previous-hour consumption levels
   have substantially larger average forecast errors in the observed data.

These findings are descriptive. They do not establish causation or prove
which model modification would improve future forecasts.

## 2. Frozen Evaluation Protocol

The following protocol must be preserved when evaluating proposed changes:

- Process the original `electricity_data.zip` directly; do not extract it.
- Require at least 54 valid minute readings per hour.
- Preserve chronological evaluation windows.
- Train only on observations available before each evaluation window.
- Use identical eligible target timestamps when comparing models.
- Keep the existing Random Forest settings frozen unless a new experiment
  is explicitly approved.
- Do not tune a model using the strict holdout results.
- Evaluate any future candidate against the same frozen baseline and
  document all changes.

Recorded Random Forest settings:

- n_estimators: 50
- max_depth: 12
- min_samples_leaf: 10
- max_features: 0.8
- n_jobs: 1
- random_state: 42

Recorded blend setting: PROFILE_WEIGHT = 0.80.

### Evaluation windows

- Window 1: 2010-04-16 22:00 to 2010-06-11 21:00
- Window 2: 2010-06-11 22:00 to 2010-08-06 21:00
- Window 3: 2010-08-06 22:00 to 2010-10-01 21:00
- Window 4: 2010-10-01 22:00 to 2010-11-26 21:00

Windows 2–4 form the strict holdout, with 3,565 shared eligible
evaluation hours.

## 3. Baseline Context

Previously recorded strict-holdout metrics:

| Method | MAE (kW) | RMSE (kW) |
|---|---:|---:|
| Previous-day baseline | 0.4824 | 0.7294 |
| Previous-week baseline | 0.5472 | 0.7937 |
| Hour-of-day profile | 0.5239 | 0.6814 |
| Frozen blend | 0.4873 | 0.6353 |
| Random Forest | 0.4251 | 0.5786 |

These values are transcribed from the existing experiment records.
The original evaluation outputs remain the authoritative source.

The Random Forest previously recorded an overall holdout bias of
+0.0631 kW, using forecast minus actual as the error convention.

## 4. Temporal-Event Diagnostic — Step 21.4

Artifact: `STEP21_4_TEMPORAL_EVENT_RESULTS.csv`

The diagnostic records the 20 largest absolute Random Forest errors and
six hours of consumption context on either side of each event.

The CSV contains 260 rows: 20 events multiplied by 13 context hours.

Examples of large errors:

| Event timestamp | Window | Actual (kW) | Forecast (kW) | Forecast error (kW) |
|---|---|---:|---:|---:|
| 2010-11-20 18:00 | 4 | 5.6268 | 2.1724 | -3.4544 |
| 2010-11-04 08:00 | 4 | 4.4189 | 1.4407 | -2.9782 |
| 2010-07-01 00:00 | 2 | 3.3290 | 0.4922 | -2.8369 |

Selected contexts showed gradual rises followed by sharp consumption
spikes and subsequent declines. Other events involved a rise that began
before the event hour.

**Interpretation:** Abrupt and evolving consumption patterns are useful
cases for further study. These examples do not prove that every large
error is caused by a sudden spike or identify a causal mechanism.

## 5. Consumption-Change Analysis — Step 21.5

Artifacts:

- `STEP21_5_CONSUMPTION_CHANGE_RECORDS.csv`
- `STEP21_5_CONSUMPTION_CHANGE_SUMMARY.csv`

Large forecast error was defined as absolute error greater than 1 kW.
Upward and downward events were classified using a window-specific
historical 90th percentile of absolute hourly changes, calculated using
pre-window history.

Recorded category summaries:

| Category | Events | MAE (kW) | RMSE (kW) | Bias (kW) | Error >1 kW |
|---|---:|---:|---:|---:|---:|
| Upward event | 133 | 0.9790 | 1.2078 | -0.9659 | 37.59% |
| Downward event | 90 | 0.5063 | 0.6024 | +0.2629 | 10.00% |
| Ordinary change | 3,342 | 0.4008 | 0.5378 | +0.0986 | 6.85% |

No unclassifiable missing-previous-hour events were recorded.

Upward events were associated with substantially higher average errors
and a negative mean forecast bias. This supports the finding that the
model has difficulty forecasting upward transitions.

## 6. Upward-Event Size Analysis — Step 21.7

| Hourly increase | Events | Mean increase (kW) | MAE (kW) | RMSE (kW) | Bias (kW) |
|---|---:|---:|---:|---:|---:|
| 1.0–<1.5 kW | 76 | 1.306 | 0.813 | 1.023 | -0.799 |
| 1.5–<2.0 kW | 38 | 1.726 | 0.873 | 1.026 | -0.855 |
| 2.0–<2.5 kW | 14 | 2.232 | 1.624 | 1.726 | -1.624 |
| At least 2.5 kW | 5 | 2.892 | 2.502 | 2.564 | -2.502 |

The largest event bands had higher average errors. However, the most
extreme band contains only five events, so its estimate is uncertain.

All reported band biases are negative. Where bias magnitude equals MAE,
the mean values indicate consistent underforecasting within that group.

Window 4 contained all five events with increases of at least 2.5 kW
and 12 of the 14 events in the 2.0–<2.5 kW band.

## 7. Previous-Hour Consumption Level — Step 21.8

Mean previous-hour consumption and event-hour results for upward events:

| Window | Events | Previous-hour level (kW) | Event actual (kW) | Forecast (kW) | Mean increase (kW) | Bias (kW) |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 25 | 0.776 | 2.233 | 1.208 | 1.457 | -1.025 |
| 3 | 30 | 0.742 | 2.152 | 1.522 | 1.410 | -0.630 |
| 4 | 78 | 1.036 | 2.726 | 1.650 | 1.690 | -1.076 |

The event-hour actual consumption was considerably higher than the
preceding-hour level in all three windows. Window 4 had the highest
mean starting level, mean increase, and absolute mean underforecast
among these window summaries.

These summaries are observational and do not isolate the independent
effect of starting consumption level from event size or other factors.

## 8. Forecast Distance Diagnostic — Step 21.9

Mean absolute distances for upward events:

| Window | Events | Distance from previous-hour actual (kW) | Distance from event-hour actual (kW) | Mean hourly increase (kW) |
|---|---:|---:|---:|---:|
| 2 | 25 | 0.606 | 1.025 | 1.457 |
| 3 | 30 | 0.852 | 0.641 | 1.410 |
| 4 | 78 | 0.794 | 1.094 | 1.690 |

The forecast was closer to the previous-hour actual on average in
Windows 2 and 4. In Window 3, it was closer to the event-hour actual.

**Conclusion:** A simple claim that forecasts always remain near the
previous-hour level is not supported. The evidence varies by window.

## 9. Correlation Diagnostics — Steps 21.10 and 21.11

The variables were hourly increase size and absolute Random Forest
forecast error.

| Group | Events | Pearson r | Spearman rho |
|---|---:|---:|---:|
| All holdout events | 133 | +0.5095 | +0.3303 |
| Window 2 | 25 | +0.5872 | +0.5554 |
| Window 3 | 30 | -0.0008 | +0.0318 |
| Window 4 | 78 | +0.5188 | +0.3638 |

Both methods show a positive association overall and in Windows 2 and 4.
Both are close to zero in Window 3.

Pearson measures linear association; Spearman measures rank-based
monotonic association. Their differences indicate that the relationship
is not equally consistent across the windows and methods.

No confidence intervals or significance tests were calculated.
Correlation does not establish causation.

## 10. Previous-Hour Level Bands — Step 21.12

| Previous-hour level | Events | Mean increase (kW) | MAE (kW) | Mean bias (kW) |
|---|---:|---:|---:|---:|
| Below 0.5 kW | 15 | 1.422 | 0.707 | -0.702 |
| 0.5–<1.0 kW | 79 | 1.575 | 0.747 | -0.726 |
| 1.0–<1.5 kW | 21 | 1.649 | 1.189 | -1.189 |
| At least 1.5 kW | 18 | 1.673 | 1.978 | -1.978 |

Average MAE was similar in the first two bands, then increased in the
higher starting-level bands. The highest band contains only 18 events.

Starting level and increase magnitude are related. This grouping alone
cannot determine whether starting level independently predicts error.

## 11. Consolidated Findings

### Supported by the observed diagnostics

1. Upward events have substantially higher MAE and RMSE than ordinary
   consumption changes.
2. The Random Forest underforecasts upward events on average.
3. Larger upward increases are positively associated with absolute
   forecast error overall, but this relationship varies by window.
4. Forecast distance comparisons do not support a universal previous-hour
   persistence explanation.
5. Higher previous-hour consumption bands have higher observed average
   error in this upward-event sample.

### Not yet established

- The causal reason for the observed forecast errors.
- Whether starting level adds predictive value after controlling for
  increase size and other features.
- Whether any proposed feature or model change improves generalization.
- Whether correlation estimates are statistically distinguishable from
  zero, particularly in smaller window samples.
- Whether diagnostic patterns will persist on other datasets or future
  time periods.

## 12. Recommended Next Experiment

Before changing the model, define a separate candidate experiment with
an explicit hypothesis and evaluation plan.

Recommended direction: investigate whether transition-related features
provide useful predictive information beyond the frozen baseline.

Requirements:

1. Keep the current Random Forest as the untouched reference.
2. Specify any new feature or modeling hypothesis before evaluation.
3. Use chronological training and evaluation only.
4. Keep holdout targets and eligibility criteria consistent.
5. Do not tune candidate settings on Windows 2–4.
6. Compare MAE, RMSE, bias, upward-event performance, and window-level
   stability.
7. Report failures and regressions as well as improvements.
8. Make no production or application-development changes until the
   experiment has been evaluated and approved.

## 13. Artifact Inventory

- `STEP21_4_TEMPORAL_EVENT_RESULTS.csv`
- `STEP21_5_CONSUMPTION_CHANGE_RECORDS.csv`
- `STEP21_5_CONSUMPTION_CHANGE_SUMMARY.csv`
- `STEP21_DIAGNOSTIC_REPORT.md` — this report

Existing scripts and source data should remain unchanged unless a
future step explicitly authorizes modifications.

---

**Final status:** Step 21 diagnostic findings consolidated. The existing
baseline remains frozen. Further experimentation requires a separate,
predefined evaluation plan.
