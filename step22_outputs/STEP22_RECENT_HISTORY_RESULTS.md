# InsightForge AI — Step 22 Recent-History Experiment

## Interpretation

Retrospective exploratory comparison on Windows 2–4; this is not independent confirmation. The primary decision uses pooled paired observations, not the unweighted average of window metrics.

## Protocol

- Reference: frozen eight-feature Random Forest. Candidate: same eight features plus seven recent-history features.
- Same shared training cohort and paired target timestamps in each window; training timestamps strictly precede each window.
- Exact prior-hour timestamps required; no imputation, gap skipping, or shortened rolling windows.
- Mean bias = forecast minus actual.
- Paired moving-block bootstrap: 10,000 replicates, 95% percentile intervals; 24h primary, 12h/48h sensitivity.
- Upward events use positive hourly increase greater than the pre-window 90th percentile of absolute consecutive-hour changes; labels remain fixed during resampling.

## Window-level counts

| Window | Ref eligible eval | Paired eval | Ref-eligible train | Shared train | Event n |
|---|---:|---:|---:|---:|---:|
| Window 2 | 1344 | 1344 | 29794 | 29746 | 25 |
| Window 3 | 966 | 966 | 31138 | 31090 | 30 |
| Window 4 | 1255 | 1255 | 32104 | 32056 | 78 |

## Per-window metrics

| Window | Model | n | MAE kW | RMSE kW | Bias kW |
|---|---|---:|---:|---:|---:|
| Window 2 | Reference | 1344 | 0.393962 | 0.506767 | 0.081611 |
| Window 2 | Candidate | 1344 | 0.294750 | 0.414967 | 0.052092 |
| Window 3 | Reference | 966 | 0.358935 | 0.487816 | 0.056620 |
| Window 3 | Candidate | 966 | 0.275483 | 0.395337 | 0.037671 |
| Window 4 | Reference | 1255 | 0.503792 | 0.696725 | 0.051248 |
| Window 4 | Candidate | 1255 | 0.369268 | 0.539499 | 0.015632 |

## Pooled paired primary results (Windows 2–4)

| Metric | Reference | Candidate | Comparison |
|---|---:|---:|---|
| Overall MAE (kW) | 0.423135 | 0.315762 | improvement 25.375% |
| Overall RMSE (kW) | 0.576355 | 0.457964 | worsening -20.541% |
| Overall bias (kW) | 0.064151 | 0.035349 | difference -0.028801 kW |
| Overall absolute-bias difference | — | — | -0.028801 kW |

Upward-event observations: **133**.
- Event MAE: reference 0.961159 kW; candidate 1.041443 kW; relative improvement -8.353%.
- Event bias: reference -0.951580 kW; candidate -1.037073 kW.
- Event absolute-bias difference 0.085494 kW; event bias difference -0.085494 kW.

## Six locked acceptance decisions

| Criterion | Point | Threshold | 95% CI | Decision |
|---|---:|---:|---|---|
| Overall MAE improvement (%) | 25.375486 | min 3 | [22.746939, 28.400630] | **PASS** |
| Overall RMSE worsening (%) | -20.541298 | max 1 | [-23.810379, -17.537347] | **PASS** |
| Overall absolute-bias difference (kW) | -0.028801 | max 0.05 | [-0.053061, -0.005384] | **PASS** |
| Upward-event MAE improvement (%) | -8.352818 | min 5 | [-14.459638, -1.772135] | **FAIL** |
| Upward-event absolute-bias difference (kW) | 0.085494 | max 0.05 | [0.023995, 0.135424] | **FAIL** |
| Upward-event bias difference (kW) | -0.085494 | min -0.05 | [-0.135424, -0.023995] | **FAIL** |

**Overall outcome: FAIL — at least one criterion failed.**

## Bootstrap sensitivity

| Block | Valid event reps | Insufficient event reps | Event reliable | Short runs/replicate |
|---:|---:|---:|---|---:|
| 12 h | 10000 | 0 | True | 1 |
| 24 h | 10000 | 0 | True | 2 |
| 48 h | 10000 | 0 | True | 2 |

Short runs use the full run as the maximum effective block; no blocks cross timestamp gaps or evaluation-window boundaries. Event-level decisions are INCONCLUSIVE if fewer than 9,500 of 10,000 replicates contain at least 10 events. Intervals are conditional on observed periods; separate intervals do not give a family-wise 95% guarantee.

## Limitations

- Windows 2–4 are retrospective exploratory periods, not independent confirmation.
- Feature importance is descriptive, not causal.
- Do not claim overall success if any criterion fails or remains inconclusive.

Full paired predictions: `step22_paired_predictions.csv`; machine-readable results: `step22_summary.json`.
