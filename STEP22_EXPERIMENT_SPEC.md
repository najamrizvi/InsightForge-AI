# InsightForge AI — Step 22 Experiment Specification

**Status:** Proposed; implementation not authorized  
**Date:** 2026-10-16  
**Experiment:** Recent-History Feature Augmentation  
**Reference model:** Existing frozen Random Forest

## 1. Objective

Determine whether predefined recent-history features improve hourly
Global Active Power forecasts compared with the existing Random Forest.

The objective is to test a hypothesis, not to assume improvement.

## 2. Hypothesis

Features describing recent consumption levels and changes may help
forecasting during upward consumption events.

The candidate must also be assessed on overall accuracy and stability.
An improvement in one subgroup is insufficient if overall performance
degrades materially.

## 3. Frozen Reference

Preserve the existing reference settings:

- n_estimators = 50
- max_depth = 12
- min_samples_leaf = 10
- max_features = 0.8
- n_jobs = 1
- random_state = 42

Do not modify the reference scripts, settings, or historical result files.

## 4. Proposed Candidate Features

Subject to inspection of the existing feature implementation, evaluate
one predefined candidate containing:

1. Previous eligible hourly active power (lag 1).
2. Active power from two hours earlier (lag 2).
3. Active power from three hours earlier (lag 3).
4. One-hour change: lag 1 minus lag 2.
5. Three-hour change: lag 1 minus lag 3.
6. Mean active power over the preceding three hours.
7. Mean active power over the preceding six hours.

All features must be calculated from observations strictly earlier
than the target hour. Rolling windows must exclude the target hour.

Use the existing eligibility policy. Do not impute missing observations
to manufacture unavailable lag values. Define feature-availability rules
before scoring and compare both models on identical eligible timestamps.

## 5. Data and Chronology

- Read the original electricity_data.zip directly.
- Do not extract or replace the source dataset.
- Require at least 54 valid minute readings per target hour.
- Preserve chronological training and evaluation.
- Train only on observations preceding each evaluation window.
- Record any candidate targets excluded because required historical
  features are unavailable.

## 6. Evaluation Design

Development should begin with a code and data-pipeline audit, followed
by a predefined development experiment using earlier chronological data.

Keep the Random Forest hyperparameters fixed for the initial comparison.
Do not select features or tune settings using Windows 2–4.

Windows 2–4 have already been extensively inspected during Step 21.
Any additional results on those windows must therefore be described as
exploratory retrospective comparisons, not untouched confirmation.

A genuinely independent confirmation requires a previously unexamined
evaluation period or suitable independent dataset.

## 7. Metrics

Report for both reference and candidate:

- MAE in kW.
- RMSE in kW.
- Mean bias, defined as forecast minus actual.
- Number of eligible evaluation hours.
- Window-level results.
- Upward-event MAE and RMSE.
- Upward-event mean bias.
- Change in overall MAE and RMSE relative to the reference.

Use identical target timestamps for paired comparisons. Report any
eligibility differences separately.

## 8. Locked Decision Rules

These thresholds are locked before the candidate experiment. They must
not be relaxed after observing candidate results.

All relative comparisons must use identical paired target timestamps.

### Overall acceptance criteria

1. Candidate MAE must improve by at least 3% relative to the reference.
2. Candidate RMSE must not worsen by more than 1% relative to the reference.
3. Candidate absolute overall mean bias must not exceed the reference
   absolute mean bias by more than 0.05 kW.

Relative improvement is calculated as:

(reference metric - candidate metric) / reference metric * 100

For RMSE, a negative improvement represents a regression.

### Upward-event acceptance criteria

4. Candidate upward-event MAE must improve by at least 5% relative to
   the reference on the same upward-event timestamps.
5. Candidate upward-event absolute mean bias must not exceed the
   reference absolute mean bias by more than 0.05 kW.
6. Candidate upward-event bias must not become more negative than the
   reference upward-event bias by more than 0.05 kW.

Upward events must be classified using the predefined historical rule:
the positive hourly consumption increase must exceed the 90th percentile
of absolute hourly changes calculated using only data before the relevant
evaluation window. The threshold must not use future observations.

### Reporting and interpretation

- Report MAE, RMSE, mean bias, sample counts, and relative changes.
- Report each evaluation window separately.
- Report upward-event MAE, RMSE, bias, and sample counts.
- Report paired uncertainty intervals, accounting for temporal
  dependence where feasible.
- Report feature-availability exclusions and any differences in
  eligible timestamps.
- Do not claim success if any acceptance criterion fails.
- If sample sizes or uncertainty make a result inconclusive, state so.
- Windows 2–4 are retrospective exploratory comparisons, not untouched
  confirmation. Independent confirmation requires a previously
  unexamined period or a suitable independent dataset.
- Keep reference hyperparameters and reference files unchanged.
- Do not begin application development as part of this experiment.

The decision rules are locked; implementation remains unauthorized
until the updated specification is reviewed and verified.
## 9. Immediate Next Action

The automated wording-presence checks for Sections 1–16 have passed. These checks do not establish full semantic or statistical correctness.

Next, manually review the complete protocol in Sections 1–16 for internal consistency, operational clarity, statistical validity, and alignment with the locked experiment decisions. Pay particular attention to the shared-training-cohort procedure, exact-timestamp feature availability, paired moving-block bootstrap, three-outcome decision rule, and upward-event bootstrap safeguard.

Do not implement or run the candidate model during this review. Preserve the reference scripts, settings, and historical results unchanged. After the manual review, explicitly decide whether to authorize the controlled retrospective candidate experiment.

W2–W4 remain exploratory retrospective data, not independent confirmation. Independent confirmation data have not yet been selected.
## 10. Current Decision

The candidate remains proposed, not implemented.
No model changes have been made.


## 11. Locked Shared-Cohort Protocol

The primary comparison will use a shared training cohort.

1. For each evaluation window, identify historical training target
   hours strictly before the window start.
2. Retain only training target hours for which both the frozen
   reference feature set and the proposed candidate feature set
   are available.
3. Train both models on exactly the same retained training timestamps.
4. Keep the reference Random Forest hyperparameters unchanged.
5. Determine the paired evaluation timestamps before scoring. Both
   models must be evaluated on exactly the same eligible target hours.
6. Apply the same predefined upward-event classification to those
   paired evaluation timestamps. Event thresholds must use only
   historical observations available before each evaluation window.
7. Report training-cohort exclusions, paired evaluation counts, and
   differences from the original reference evaluation wherever relevant.
8. Preserve the original reference scripts, settings, and historical
   result files unchanged.

This shared-cohort protocol is locked for the primary comparison.
### Primary acceptance population

- The primary acceptance decision uses the pooled paired evaluation
  observations from Windows 2, 3, and 4 combined.
- Calculate pooled metrics from the combined paired observations;
  do not take an unweighted average of window-level metrics.
- Apply all six locked acceptance criteria and their corresponding
  uncertainty-bound rules to the pooled results.
- Report each window separately as a secondary descriptive analysis.
  Window-level results do not replace the predefined pooled primary
  decision, but material regressions or instability must be disclosed.
- Apply upward-event criteria to the predefined upward-event subset
  of the same paired evaluation observations.
- These Windows 2–4 results remain retrospective exploratory evidence,
  not independent confirmation. A pooled pass does not establish
  independent generalization.

It does not authorize candidate implementation or model execution.
Implementation remains unauthorized until the specification and
audit protocol have been reviewed and verified.

## 12. Locked Exact-Timestamp Feature Availability

Candidate lag and rolling features must use exact preceding hourly
timestamps from the eligible hourly dictionary.

- Lag 1, lag 2, and lag 3 require their exact preceding hourly timestamps.
- The three-hour mean requires all three preceding hourly timestamps.
- The six-hour mean requires all six preceding hourly timestamps.
- Do not skip missing hours, impute missing observations, or calculate
  shorter or irregular rolling windows.
- If any required timestamp is unavailable, the feature row is
  unavailable and must be excluded from the shared training cohort
  or paired evaluation set, as applicable.
- Apply these rules consistently before fitting or scoring either model.

This rule is locked. It does not authorize candidate implementation
or model execution.

## 13. Development and Independent Confirmation

### Retrospective development

- Windows 2–4 have already been examined during prior diagnostics.
- Results from these windows are retrospective exploratory comparisons.
- Do not describe these results as untouched or independent confirmation.
- Apply the locked feature, training-cohort, timestamp, and acceptance
  rules without relaxing them after observing candidate results.

### Independent confirmation

- Independent confirmation requires a previously unexamined evaluation
  period or a suitable independent dataset.
- Define and document the confirmation data and evaluation procedure
  before examining its model results.
- Freeze the candidate feature definition and model settings before
  independent confirmation.
- Do not tune features, thresholds, or hyperparameters against the
  confirmation results.
- Report the confirmation metrics, sample counts, uncertainty, and
  limitations separately from retrospective development results.

The independent confirmation data have not yet been selected.
This section does not authorize candidate implementation or model
execution.

## 14. Locked Paired Uncertainty Analysis

### Bootstrap design

- Use a paired moving-block bootstrap with 10,000 replicates.
- Use 95% confidence intervals, with percentile interval endpoints
  at the 2.5th and 97.5th percentiles.
- Use 24 consecutive hourly observations as the primary block length.
- Repeat the analysis using 12-hour and 48-hour blocks as sensitivity
  checks.
- Resample paired reference and candidate records together, preserving
  their timestamp alignment.
- Keep each evaluation window separate during block resampling.
  No block may cross an evaluation-window boundary.
- Preserve each window's original evaluation sample count per replicate.

### Operational block-sampling procedure

- Within each evaluation window, divide the final paired evaluation
  cohort into maximal runs whose timestamps are exactly one hour apart.
- Use non-circular moving blocks sampled with replacement from within
  each run. Never cross a timestamp gap or evaluation-window boundary.
- Choose block start positions uniformly from all valid starts that
  keep the entire block within its source run.
- For each run, concatenate sampled blocks and truncate the final block
  as needed to reproduce that run's original observation count.
- A block cannot be longer than its source run. If a run is shorter
  than the requested 12-, 24-, or 48-hour block length, use the entire
  run as the maximum available block and report this limitation and
  the affected observation counts for each block-length analysis.
- Resample paired reference/candidate records and their fixed event
  labels together. Apply the same sampled indices to all paired values.
- Report the requested block length and any shorter effective blocks.
  Do not join separated runs to manufacture longer blocks.

### Quantities to report

Estimate paired uncertainty intervals for:
- Candidate-minus-reference MAE.
- Candidate-minus-reference RMSE.
- Candidate-minus-reference mean bias.
- Difference in absolute mean bias.
- Relative MAE improvement.
- Relative RMSE worsening.
- Upward-event relative MAE improvement.
- Upward-event MAE and bias comparisons on the predefined event subset.

Keep the predefined upward-event labels fixed during resampling.
Report event sample counts and explain limitations arising from
the smaller event subgroup.

### Interpretation

- Report the primary 24-hour results and compare them with the
  12-hour and 48-hour sensitivity analyses.
- Explain any material changes in the conclusions across block lengths.
- Treat intervals as conditional on the observed evaluation periods;
  they do not establish independent generalization.
- Do not claim that an acceptance criterion is conclusively satisfied
  when the point estimate passes but uncertainty leaves the conclusion
  unresolved.
- Do not relax the locked acceptance thresholds after observing results.

This uncertainty protocol is locked. Candidate implementation and
model execution remain unauthorized pending final specification review.

## 15. Locked Three-Outcome Decision Rule

Evaluate each of the six acceptance criteria separately using paired
point estimates and the 95% confidence intervals defined in Section 14.

For the confidence-bound tests below, define:

- Relative MAE improvement (%) =
  (reference MAE - candidate MAE) / reference MAE * 100.
- Relative RMSE worsening (%) =
  (candidate RMSE - reference RMSE) / reference RMSE * 100.
- Overall absolute-bias difference (kW) =
  abs(candidate overall bias) - abs(reference overall bias).
- Upward-event absolute-bias difference (kW) =
  abs(candidate upward-event bias) - abs(reference upward-event bias).
- Upward-event bias difference (kW) =
  candidate upward-event bias - reference upward-event bias.

Calculate each applicable quantity within every paired bootstrap
replicate. In particular, calculate relative MAE improvement and
relative RMSE worsening within each replicate rather than deriving
their intervals from separate metric intervals.

A PASS requires both the point estimate and its 95% confidence interval
to support the locked threshold:

1. Overall MAE: point relative improvement >= 3%, and the lower
   confidence bound for relative MAE improvement is >= 3%.
2. Overall RMSE: point relative worsening <= 1%, and the upper
   confidence bound for relative RMSE worsening is <= 1%.
3. Overall absolute bias: point absolute-bias difference <= 0.05 kW,
   and its upper confidence bound is <= 0.05 kW.
4. Upward-event MAE: point relative improvement >= 5%, and the lower
   confidence bound for relative improvement is >= 5%.
5. Upward-event absolute bias: point absolute-bias difference
   <= 0.05 kW, and its upper confidence bound is <= 0.05 kW.
6. Upward-event bias direction: point bias difference >= -0.05 kW,
   and its lower confidence bound is >= -0.05 kW.

Apply the following three-outcome rule to each criterion:

- PASS: the point estimate meets the threshold and the relevant
  confidence bound also supports meeting it.
- FAIL: the point estimate violates the locked threshold, or the
  interval provides evidence of a meaningful violation under the
  criterion's stated direction.
- INCONCLUSIVE: the point estimate meets the threshold, but the
  confidence interval still includes unacceptable outcomes and does
  not support PASS.

For event-level criteria, an unreliable event bootstrap under Section 16
requires an INCONCLUSIVE result regardless of the point estimate.

Report each criterion separately. Passing individual intervals does
not establish a formal family-wise 95% guarantee across all criteria.
Do not claim overall experiment success if any required criterion
fails or remains inconclusive.

All six acceptance thresholds remain locked. This clarification defines
how to assess their confidence-bound support; it does not relax them.
It does not authorize implementation or model execution.
## 16. Locked Upward-Event Bootstrap Safeguard

- Use the paired moving-block bootstrap specified in Section 14 for
  overall metrics and preserve paired reference/candidate timestamps.
- For upward-event metrics, keep the historically defined event labels
  fixed and preserve the paired timestamp sequence.
- Resample contiguous hourly blocks using the operational procedure
  defined in Section 14. Do not cross timestamp gaps or evaluation-
  window boundaries.
- A bootstrap replicate is valid for event-level metrics only if it
  contains at least 10 upward-event observations in the relevant
  event subset. Apply this minimum consistently to the paired event
  metrics for that replicate.
- Of the 10,000 planned replicates, at least 95% (at least 9,500)
  must be valid for the event-level uncertainty analysis to be treated
  as sufficiently reliable under this protocol.
- Report the number and proportion of valid and insufficient-event
  replicates. Do not silently discard insufficient-event replicates.
  If event intervals are calculated from valid replicates, explicitly
  report that fact and the excluded count.
- If fewer than 9,500 replicates are valid, or the event interval is
  otherwise unreliable, mark the affected event-level decision
  INCONCLUSIVE. Do not present an unreliable interval as evidence
  for PASS or FAIL.
- Report event sample counts and explain limitations from the smaller
  event subgroup. This safeguard does not replace the locked acceptance
  thresholds or the three-outcome rule in Section 15.

These reliability rules are fixed before any candidate experiment.
They do not authorize implementation or model execution.




