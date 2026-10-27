# InsightForge AI
### Evidence-Driven Household Electricity Analytics & Forecasting

InsightForge AI is an interactive data analytics and machine-learning project built around household electricity consumption data. It brings consumption exploration, forecasting evaluation, model comparisons, and reliability limitations into one Streamlit dashboard.

The project emphasizes **reproducible evaluation and honest reporting**: a model is not treated as a winner simply because its average error improves. Performance on predefined upward-consumption events and the limitations of the available evidence are reported alongside overall metrics.

> **Project status:** The local dashboard has been implemented and visually checked. Forecast experiments and results are documented. The recent-history candidate improved overall error but did **not** pass the predefined upward-event decision gate, so it remains exploratory rather than replacing the reference model.

## Contents
- [Overview](#project-overview)
- [Dashboard Features](#dashboard-features)
- [Dataset](#dataset)
- [Data Quality and Preparation](#data-quality-and-preparation)
- [Forecasting Methodology](#forecasting-methodology)
- [Results](#results)
- [Interpretation and Decision](#interpretation-and-decision)
- [Technology Stack](#technology-stack)
- [Run Locally](#run-locally)
- [Project Structure](#project-structure)
- [Reproducibility and Safeguards](#reproducibility-and-evaluation-safeguards)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Data Source and Attribution](#data-source-and-attribution)

## Project Overview

Household electricity demand changes with the time of day, day of the week, season, and recent consumption behavior. A useful forecasting workflow needs more than a prediction: it also needs historical context, comparisons against simple baselines, and transparent evidence about where predictions fail.

InsightForge AI demonstrates practical data analysis, time-series feature engineering, model evaluation, visualization, and responsible interpretation of machine-learning results.

### Goals
- Explore long-term and time-based electricity consumption patterns.
- Compare forecasting baselines with a Random Forest model.
- Evaluate forecasts on later time periods separated chronologically from training.
- Examine large errors and performance on predefined upward-consumption events.
- Present findings and limitations in an interactive dashboard.
- Keep exploratory findings distinct from conclusions that pass predefined decision criteria.

## Dashboard Features

The Streamlit application contains five sections:

| Section | Purpose |
|---|---|
| **Overview** | Summarizes the project and its main findings. |
| **Consumption Explorer** | Explores electricity consumption over time and across time-based patterns. |
| **Forecast Analysis** | Examines actual versus predicted consumption and forecast errors. |
| **Model Comparison** | Compares forecasting models using error metrics and evaluation windows. |
| **Reliability & Limitations** | Explains data-quality constraints, model behavior, and interpretation limits. |

## Dataset

**Dataset:** Individual Household Electric Power Consumption  
**Publisher:** UCI Machine Learning Repository  
**Source:** [UCI dataset page](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

The dataset contains measurements from a household electricity monitoring system at approximately one-minute intervals.

| Property | Details |
|---|---|
| Raw records | 2,075,259 |
| Raw fields | 9 |
| Date range | 16 December 2006 – 26 November 2010 |
| Main forecasting target | `Global_active_power` |
| Original measurement cadence | Approximately one minute |
| Forecasting resolution | Hourly |
| Source archive | `electricity_data.zip` |

The raw data includes date and time fields, global active and reactive power, voltage, global intensity, and three sub-metering measurements.

## Data Quality and Preparation

The data audit recorded:
- 2,075,259 source records across nine columns.
- 25,979 rows with missing measurement values.
- No duplicate timestamps detected.
- No out-of-order timestamps detected.
- No missing minute intervals detected in the audited sequence.
- Hourly targets are eligible only when at least **54 valid minute readings** are available.
- 34,085 complete hours, 83 partial hours, and 421 hours with no valid readings.

The experiment reads the source data directly from the ZIP archive; the original archive is not modified or extracted as part of the documented workflow.

The minimum-reading rule helps prevent a partially observed hour from being silently treated as a complete hour.

## Forecasting Methodology

### Chronological evaluation

Evaluation uses chronological windows. Each model is assessed on future timestamps, with training data restricted to timestamps before the applicable evaluation window. This reduces temporal leakage and better reflects a forecasting scenario.

| Window | Evaluation period | Eligible hours |
|---|---|---:|
| W1 | 16 Apr 2010 22:00 – 11 Jun 2010 21:00 | 1,344 |
| W2 | 11 Jun 2010 22:00 – 6 Aug 2010 21:00 | 1,344 |
| W3 | 6 Aug 2010 22:00 – 1 Oct 2010 21:00 | 966 |
| W4 | 1 Oct 2010 22:00 – 26 Nov 2010 21:00 | 1,255 |

The strict later-window comparison uses W2–W4 and **3,565 evaluation hours**.

### Models compared
- **Previous-day baseline:** uses consumption from the same hour on the previous day.
- **Previous-week baseline:** uses consumption from the same hour one week earlier.
- **Hour-of-day profile:** forecasts using a historical hour-of-day profile.
- **Frozen blend:** combines the previous-day forecast with the hour-of-day profile. Its weight was selected on W1 and frozen at 0.80 for later evaluation.
- **Random Forest:** a supervised model using historical lags and calendar-derived features.
- **Recent-history candidate:** an experimental feature set adding recent lags and rolling-history information under strict timestamp-availability rules.

### Random Forest configuration

| Parameter | Value |
|---|---:|
| `n_estimators` | 50 |
| `max_depth` | 12 |
| `min_samples_leaf` | 10 |
| `max_features` | 0.8 |
| `n_jobs` | 1 |
| `random_state` | 42 |

The baseline Random Forest uses lagged consumption, including 24-hour and 168-hour lags, and calendar features. The recent-history candidate requires real preceding hourly timestamps: missing timestamps are not skipped, and missing history is not imputed to manufacture features.

## Results

### Frozen model comparison — strict holdout W2–W4

All models are compared on the same 3,565 later-window evaluation hours. Lower MAE and RMSE indicate smaller forecast errors.

| Model | MAE (kW) | RMSE (kW) |
|---|---:|---:|
| **Random Forest** | **0.4251** | **0.5786** |
| Previous-day baseline | 0.4824 | 0.7294 |
| Frozen blend | 0.4873 | 0.6353 |
| Hour-of-day profile | 0.5239 | 0.6814 |
| Previous-week baseline | 0.5472 | 0.7937 |

The Random Forest achieved the lowest MAE and RMSE among these frozen models. The frozen blend improved RMSE relative to the previous-day baseline, but did not improve MAE.

The Random Forest's pooled result across all four evaluation windows was MAE **0.4316 kW** and RMSE **0.5850 kW**. The pooled result is distinct from the strict W2–W4 holdout result.

### Recent-history candidate — paired evaluation

The candidate was compared with its reference on the same 3,565 paired timestamps.

| Metric | Reference | Recent-history candidate |
|---|---:|---:|
| Overall MAE | 0.4231 kW | 0.3158 kW |
| Overall RMSE | 0.5764 kW | 0.4580 kW |
| Signed bias | +0.0642 kW | +0.0353 kW |
| Upward-event MAE (133 observations) | 0.9612 kW | 1.0414 kW |

The candidate reduced overall MAE by approximately **25.38%** and overall RMSE by approximately **20.54%** relative to its paired reference. However, its MAE on the predefined upward-event subset was approximately **8.35% worse**: 1.0414 kW versus 0.9612 kW.

The overall metrics indicate better average performance across paired timestamps. The upward-event result indicates worse performance on a specific subset that may be important for understanding rapid increases in consumption.

## Interpretation and Decision

**Decision: the recent-history candidate does not pass the predefined Step 22 decision gate.**

Because upward-event MAE worsened, the candidate is **not promoted as a full replacement** for the reference model. It remains exploratory.

The experiment uses paired moving-block bootstrap confidence intervals with 10,000 replicates and 12-hour, 24-hour, and 48-hour block settings to account for temporal dependence in paired errors. The resulting artifacts are saved; the dashboard should display saved results rather than silently recomputing the experiment.

This is a meaningful finding, not a failed project: it demonstrates why model selection should consider predefined decision rules, subgroup performance, and trade-offs—not overall MAE alone.

## Technology Stack

- **Python** — data processing and experiment implementation
- **Pandas** — time-series preparation and analysis
- **NumPy** — numerical computation
- **scikit-learn** — Random Forest model and machine-learning utilities
- **Streamlit** — interactive web dashboard
- **Plotly** — interactive charts and visualizations

## Run Locally

### Prerequisites
- Python 3.12 or a compatible Python version
- The original `electricity_data.zip` dataset archive
- Git, if cloning the repository

### 1. Open the project folder

In Windows PowerShell:

```powershell
cd "C:\Users\Administrator\Documents\InsightForge-Experiment"
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If a `requirements.txt` file is present:

```powershell
python -m pip install -r .\requirements.txt
```

If the project does not yet have a requirements file, install the direct dashboard dependencies in the active environment:

```powershell
python -m pip install streamlit pandas numpy scikit-learn plotly
```

For a reproducible setup, maintain a tested `requirements.txt` in the repository.

### 4. Launch the dashboard

```powershell
python -m streamlit run .\app.py
```

Streamlit prints a local URL in the terminal, commonly `http://localhost:8501`. Open it in your browser.

### 5. Stop the dashboard

Press `Ctrl+C` in the PowerShell terminal.

## Project Structure

The following is a documentation-oriented outline of core files and saved results; the local folder may contain additional experiment scripts and audit artifacts.

```text
InsightForge-Experiment/
├── app.py
├── electricity_data.zip
├── STEP18_FORECASTING_RESULTS.md
├── STEP22_EXPERIMENT_SPEC.md
└── step22_outputs/
    ├── STEP22_RECENT_HISTORY_RESULTS.md
    ├── step22_summary.json
    └── step22_paired_predictions.csv
```

- `app.py` — Streamlit dashboard.
- `electricity_data.zip` — source dataset archive used by the experiment.
- `STEP18_FORECASTING_RESULTS.md` — frozen baseline results.
- `STEP22_EXPERIMENT_SPEC.md` — specification and evaluation protocol for the recent-history experiment.
- `step22_outputs/STEP22_RECENT_HISTORY_RESULTS.md` — saved candidate evaluation report.
- `step22_outputs/step22_summary.json` — machine-readable experiment summary.
- `step22_outputs/step22_paired_predictions.csv` — paired predictions for candidate comparison.

Keep the frozen experiment specifications and result artifacts intact. Dashboard presentation changes should not modify the original evaluation outputs.

## Reproducibility and Evaluation Safeguards

1. **Chronological evaluation:** training data precedes each evaluation window.
2. **Shared evaluation timestamps:** comparisons use aligned timestamps.
3. **Frozen baseline settings:** baseline choices are not repeatedly changed after observing later-window results.
4. **No invented lag history:** the candidate requires exact preceding timestamps and does not skip gaps or impute missing lags.
5. **Paired uncertainty analysis:** block bootstrap analysis accounts for temporal dependence in paired errors.
6. **Predefined decision gate:** upward-event performance is considered alongside overall performance.
7. **Saved results:** the dashboard is intended to display documented outputs rather than alter frozen experiments.
8. **Transparent reporting:** both the candidate's overall improvement and its upward-event regression are disclosed.

## Limitations

- The dataset represents one household; findings should not be assumed to generalize to other households, buildings, or electricity systems.
- Historical patterns may change due to behavior, appliances, weather, occupancy, and factors not fully represented in the dataset.
- The hourly-validity rule reduces incomplete-hour problems but does not eliminate all measurement uncertainty.
- Lower average error does not guarantee reliable performance during rapid upward consumption changes.
- The upward-event subset contains 133 observations, so conclusions about it should be interpreted with the limited sample size in mind.
- Forecasts are estimates, not guarantees, and should not be the sole basis for high-stakes operational decisions.
- This is an analytical portfolio project, not a production energy-management or safety-critical forecasting service.

## Future Improvements

Potential next steps, subject to a separate scope decision:
- Add dashboard links to methodology and saved experiment reports.
- Capture representative screenshots and include them in the repository.
- Add automated smoke checks for application startup and key data-loading paths.
- Validate a pinned `requirements.txt` in a clean environment.
- Publish the repository and consider deployment after confirming data-file handling and hosting constraints.
- Investigate upward-event behavior only as a separately scoped future experiment, without retroactively changing the Step 22 decision.

## Data Source and Attribution

This project uses the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository.

- Dataset page: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- The UCI dataset record identifies the dataset as distributed under **CC BY 4.0**. Consult the official page for its recommended citation and attribution details when redistributing the data.

This is an independent educational and portfolio analysis and is not affiliated with or endorsed by UCI.

---

## Author and Links

**Project:** InsightForge AI — Evidence-Driven Household Electricity Analytics & Forecasting

- **GitHub repository:** Add the public repository URL after publication.
- **Live dashboard:** Add a deployed application URL only after deployment is completed and verified.
- **Portfolio screenshots:** Add links after capturing screenshots from the final dashboard.

*Keep all repository, deployment, and screenshot links accurate. Do not claim a public deployment until it is live and verified.*
