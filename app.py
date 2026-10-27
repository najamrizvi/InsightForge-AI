
from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# InsightForge AI
# Evidence-first household electricity forecasting dashboard
# ============================================================

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "electricity_data.zip"
OUTPUT_DIR = ROOT / "step22_outputs"
PREDICTIONS_PATH = OUTPUT_DIR / "step22_paired_predictions.csv"
SUMMARY_PATH = OUTPUT_DIR / "step22_summary.json"

WINDOW_ORDER = ["Window 2", "Window 3", "Window 4"]

st.set_page_config(
    page_title="InsightForge AI | Forecast Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Presentation ----------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F7F9FC;
    }
    [data-testid="stSidebar"] {
        background-color: #102A43;
    }
    [data-testid="stSidebar"] * {
        color: #F5F7FA;
    }
    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 16px;
        background: linear-gradient(115deg, #102A43, #087F8C);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        color: white;
        font-size: 2.15rem;
        margin-bottom: 0.35rem;
    }
    .hero p {
        color: #E5F4F6;
        font-size: 1rem;
        margin-bottom: 0.2rem;
    }
    .section-note {
        color: #526579;
        font-size: 0.92rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E1E8EF;
        border-radius: 12px;
        padding: 14px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Shared helpers ----------

def show_hero(subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>InsightForge AI</h1>
            <p>Evidence-first forecasting and consumption analytics</p>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def check_required_files() -> list[str]:
    missing = []
    for path in (ZIP_PATH, PREDICTIONS_PATH, SUMMARY_PATH):
        if not path.is_file():
            missing.append(str(path.relative_to(ROOT)))
    return missing


@st.cache_data(show_spinner=False)
def load_saved_results() -> tuple[pd.DataFrame, dict]:
    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["timestamp"],
    )

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    required_columns = {
        "window",
        "timestamp",
        "actual",
        "reference_prediction",
        "candidate_prediction",
        "is_upward_event",
    }
    missing = required_columns.difference(predictions.columns)

    if missing:
        raise ValueError(
            "The saved prediction file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    predictions = predictions.sort_values("timestamp").reset_index(drop=True)
    return predictions, summary


@st.cache_data(show_spinner="Reading the household dataset from the ZIP archive...")
def load_hourly_history() -> pd.DataFrame:
    """
    Read only the date, time, and active-power fields.
    The source ZIP is read in place and is never extracted or modified.
    Eligible hours require at least 54 valid minute readings.
    """
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        members = [
            name
            for name in archive.namelist()
            if name.lower().endswith("household_power_consumption.txt")
        ]

        if len(members) != 1:
            raise RuntimeError(
                "Expected exactly one household_power_consumption.txt "
                f"file in the ZIP; found {len(members)}."
            )

        with archive.open(members[0], "r") as binary:
            with io.TextIOWrapper(
                binary,
                encoding="utf-8",
                newline="",
            ) as stream:
                raw = pd.read_csv(
                    stream,
                    sep=";",
                    usecols=["Date", "Time", "Global_active_power"],
                    na_values=["?"],
                    dtype={
                        "Date": "string",
                        "Time": "string",
                        "Global_active_power": "float64",
                    },
                    low_memory=False,
                )

    raw["timestamp"] = pd.to_datetime(
        raw["Date"] + " " + raw["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    raw = raw.dropna(subset=["timestamp", "Global_active_power"]).copy()
    raw["hour"] = raw["timestamp"].dt.floor("h")

    hourly = (
        raw.groupby("hour")["Global_active_power"]
        .agg(mean_kw="mean", valid_minutes="count")
        .reset_index()
    )

    hourly = hourly.loc[hourly["valid_minutes"] >= 54].copy()
    hourly = hourly.sort_values("hour").reset_index(drop=True)

    hourly["date"] = hourly["hour"].dt.date
    hourly["hour_of_day"] = hourly["hour"].dt.hour
    hourly["weekday"] = hourly["hour"].dt.day_name()
    hourly["month"] = hourly["hour"].dt.to_period("M").astype(str)

    return hourly


def metric_frame(frame: pd.DataFrame, actual_col: str, pred_col: str) -> dict:
    pair = frame[[actual_col, pred_col]].dropna()

    if pair.empty:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
        }

    error = pair[pred_col].to_numpy() - pair[actual_col].to_numpy()

    return {
        "n": len(pair),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


def improvement_percent(reference: float, candidate: float) -> float:
    if not np.isfinite(reference) or reference == 0:
        return np.nan
    return 100.0 * (reference - candidate) / reference


def get_window_order(frame: pd.DataFrame) -> list[str]:
    found = frame["window"].dropna().astype(str).unique().tolist()
    ordered = [w for w in WINDOW_ORDER if w in found]
    return ordered + [w for w in found if w not in ordered]


def format_kw(value: float) -> str:
    return "—" if not np.isfinite(value) else f"{value:.4f} kW"


def format_pct(value: float) -> str:
    return "—" if not np.isfinite(value) else f"{value:+.2f}%"


def flatten_json(value, prefix=""):
    """Flatten saved JSON for optional inspection without assuming its schema."""
    rows = []

    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_json(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            rows.extend(flatten_json(child, path))
    else:
        rows.append({"field": prefix, "value": str(value)})

    return rows


def safe_plotly(fig, height=390):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=45, b=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#243B53"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E8EEF4")
    st.plotly_chart(fig, use_container_width=True)


# ---------- Startup checks ----------

missing_files = check_required_files()

if missing_files:
    st.error("Some required project files could not be found.")
    for item in missing_files:
        st.write(f"- `{item}`")
    st.info(
        "Keep app.py in the project root and preserve the existing "
        "step22_outputs folder and dataset ZIP."
    )
    st.stop()

try:
    predictions, summary = load_saved_results()
except Exception as exc:
    st.error(f"Could not load the saved Step 22 results: {exc}")
    st.stop()

if predictions.empty:
    st.error("The saved prediction file contains no rows.")
    st.stop()


# ---------- Navigation ----------

with st.sidebar:
    st.markdown("## ⚡ InsightForge AI")
    st.caption("Forecast analytics workspace")
    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Consumption Explorer",
            "Forecast Analysis",
            "Model Comparison",
            "Reliability & Limitations",
        ],
    )
    st.divider()
    st.caption("Dataset: UCI Individual Household Electric Power Consumption")
    st.caption("Source archive is read-only.")
    st.caption("Forecasts shown here are saved experimental results.")


# ---------- Shared data ----------

reference_metrics = metric_frame(
    predictions, "actual", "reference_prediction"
)
candidate_metrics = metric_frame(
    predictions, "actual", "candidate_prediction"
)

overall_mae_improvement = improvement_percent(
    reference_metrics["mae"], candidate_metrics["mae"]
)
overall_rmse_improvement = improvement_percent(
    reference_metrics["rmse"], candidate_metrics["rmse"]
)

event_rows = predictions.loc[
    predictions["is_upward_event"].fillna(0).astype(int) == 1
].copy()

reference_event_metrics = metric_frame(
    event_rows, "actual", "reference_prediction"
)
candidate_event_metrics = metric_frame(
    event_rows, "actual", "candidate_prediction"
)

windows = get_window_order(predictions)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "Overview":
    show_hero("A transparent view of household consumption and forecasting evidence")

    st.markdown(
        """
        This workspace combines historical electricity measurements with
        chronological forecasting results. It distinguishes overall forecasting
        accuracy from performance during rapid upward-consumption events.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Paired forecast hours", f"{len(predictions):,}")
    col2.metric("Evaluation windows", f"{len(windows)}")
    col3.metric(
        "Candidate MAE improvement",
        format_pct(overall_mae_improvement),
    )
    col4.metric(
        "Upward-event observations",
        f"{len(event_rows):,}",
    )

    st.subheader("Forecast performance at a glance")

    compare = pd.DataFrame(
        [
            {
                "Model": "Frozen reference",
                "MAE (kW)": reference_metrics["mae"],
                "RMSE (kW)": reference_metrics["rmse"],
            },
            {
                "Model": "Step 22 candidate",
                "MAE (kW)": candidate_metrics["mae"],
                "RMSE (kW)": candidate_metrics["rmse"],
            },
        ]
    )

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            compare,
            x="Model",
            y="MAE (kW)",
            title="Mean absolute error",
            text_auto=".3f",
        )
        fig.update_traces(marker_color="#087F8C")
        safe_plotly(fig, 330)

    with right:
        fig = px.bar(
            compare,
            x="Model",
            y="RMSE (kW)",
            title="Root mean squared error",
            text_auto=".3f",
        )
        fig.update_traces(marker_color="#2874A6")
        safe_plotly(fig, 330)

    st.subheader("Evidence summary")

    st.markdown(
        f"""
        - The candidate's paired overall MAE is **{format_kw(candidate_metrics["mae"])}**,
          compared with **{format_kw(reference_metrics["mae"])}** for the reference.
        - Overall MAE improvement: **{format_pct(overall_mae_improvement)}**.
        - Upward-event MAE: **{format_kw(candidate_event_metrics["mae"])}** for the
          candidate versus **{format_kw(reference_event_metrics["mae"])}** for the reference.
        - The candidate did **not** pass the complete predefined Step 22 decision
          protocol. Better pooled accuracy does not establish that it is a safe
          full replacement.
        """
    )

    st.info(
        "These are results from one household electricity dataset and the saved "
        "chronological evaluation. They do not establish general performance "
        "for other households or guarantee future accuracy."
    )

    with st.expander("Project and experiment metadata"):
        st.write("Experiment:", summary.get("experiment", "Not specified"))
        st.write("Saved status:", summary.get("status", "Not specified"))
        st.write("Overall decision:", summary.get("overall_outcome", "Not specified"))
        st.write("Reference script:", summary.get("reference_script", "Not specified"))
        st.write("Evaluation windows:", summary.get("evaluation_windows", "Not specified"))


# ============================================================
# PAGE 2 — CONSUMPTION EXPLORER
# ============================================================

elif page == "Consumption Explorer":
    show_hero("Explore historical active-power readings")

    st.markdown(
        """
        The historical explorer reads the original ZIP directly. It uses the
        same minimum of 54 valid minute readings per eligible hour. Historical
        active power is reported in kW; it is not a claim about a different
        household or a future forecast.
        """
    )

    try:
        history = load_hourly_history()
    except Exception as exc:
        st.error(f"Could not process the historical dataset: {exc}")
        st.stop()

    if history.empty:
        st.warning("No eligible hourly readings were found.")
        st.stop()

    minimum_date = history["hour"].min().date()
    maximum_date = history["hour"].max().date()

    date_col1, date_col2 = st.columns(2)

    with date_col1:
        start_date = st.date_input(
            "Start date",
            value=minimum_date,
            min_value=minimum_date,
            max_value=maximum_date,
            key="history_start",
        )

    with date_col2:
        end_date = st.date_input(
            "End date",
            value=maximum_date,
            min_value=minimum_date,
            max_value=maximum_date,
            key="history_end",
        )

    if start_date > end_date:
        st.error("The start date must be on or before the end date.")
        st.stop()

    selected = history.loc[
        (history["date"] >= start_date)
        & (history["date"] <= end_date)
    ].copy()

    if selected.empty:
        st.warning("No eligible readings are available for this date range.")
        st.stop()

    a, b, c, d = st.columns(4)
    a.metric("Eligible hours", f"{len(selected):,}")
    b.metric("Average active power", format_kw(selected["mean_kw"].mean()))
    c.metric("Peak hourly mean", format_kw(selected["mean_kw"].max()))
    d.metric("Days represented", f"{selected['date'].nunique():,}")

    daily = (
        selected.groupby("date", as_index=False)
        .agg(
            mean_active_power_kw=("mean_kw", "mean"),
            eligible_hours=("mean_kw", "count"),
        )
    )

    fig = px.line(
        daily,
        x="date",
        y="mean_active_power_kw",
        markers=False,
        title="Daily average of eligible hourly active-power readings",
        labels={
            "date": "Date",
            "mean_active_power_kw": "Average active power (kW)",
        },
    )
    fig.update_traces(line=dict(color="#087F8C", width=2))
    safe_plotly(fig, 400)

    hourly_profile = (
        selected.groupby("hour_of_day", as_index=False)
        .agg(mean_kw=("mean_kw", "mean"))
    )

    fig = px.bar(
        hourly_profile,
        x="hour_of_day",
        y="mean_kw",
        title="Average active power by hour of day",
        labels={
            "hour_of_day": "Hour of day",
            "mean_kw": "Average active power (kW)",
        },
    )
    fig.update_traces(marker_color="#2874A6")
    safe_plotly(fig, 350)

    monthly = (
        selected.groupby("month", as_index=False)
        .agg(
            mean_kw=("mean_kw", "mean"),
            eligible_hours=("mean_kw", "count"),
        )
    )

    fig = px.bar(
        monthly,
        x="month",
        y="mean_kw",
        title="Monthly average active power in the selected period",
        labels={"month": "Month", "mean_kw": "Average active power (kW)"},
    )
    fig.update_traces(marker_color="#168AAD")
    safe_plotly(fig, 350)

    with st.expander("View hourly records"):
        st.dataframe(
            selected[["hour", "mean_kw", "valid_minutes"]].rename(
                columns={
                    "hour": "Hour",
                    "mean_kw": "Mean active power (kW)",
                    "valid_minutes": "Valid minute readings",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Only hours with at least 54 valid minute readings are included. "
        "Daily and monthly averages are calculated from eligible hours, so "
        "they may not represent every minute of each day."
    )


# ============================================================
# PAGE 3 — FORECAST ANALYSIS
# ============================================================

elif page == "Forecast Analysis":
    show_hero("Compare saved forecasts with actual hourly readings")

    chosen_window = st.selectbox(
        "Evaluation window",
        options=["All evaluation windows"] + windows,
    )

    if chosen_window == "All evaluation windows":
        frame = predictions.copy()
    else:
        frame = predictions.loc[predictions["window"] == chosen_window].copy()

    frame = frame.sort_values("timestamp")

    if frame.empty:
        st.warning("No saved predictions are available for this selection.")
        st.stop()

    actual_mean = frame["actual"].mean()
    ref_m = metric_frame(frame, "actual", "reference_prediction")
    cand_m = metric_frame(frame, "actual", "candidate_prediction")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Evaluation hours", f"{len(frame):,}")
    m2.metric("Actual mean active power", format_kw(actual_mean))
    m3.metric("Reference MAE", format_kw(ref_m["mae"]))
    m4.metric("Candidate MAE", format_kw(cand_m["mae"]))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["actual"],
            name="Actual",
            mode="lines",
            line=dict(color="#102A43", width=1.8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["reference_prediction"],
            name="Frozen reference",
            mode="lines",
            line=dict(color="#2874A6", width=1.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["candidate_prediction"],
            name="Step 22 candidate",
            mode="lines",
            line=dict(color="#E09F3E", width=1.2),
        )
    )
    fig.update_layout(
        title=dict(
            text="Actual vs. predicted consumption",
            font=dict(size=18),
            automargin=True,
        ),
        title_pad=dict(t=18, b=12),
        xaxis_title="Timestamp",
        yaxis_title="Active power (kW)",
        margin=dict(t=85),
    )
    safe_plotly(fig, 460)

    st.subheader("Forecast errors over time")

    error_frame = frame[
        ["timestamp", "reference_error", "candidate_error"]
    ].copy()

    if "reference_error" not in error_frame.columns:
        error_frame["reference_error"] = (
            frame["reference_prediction"] - frame["actual"]
        )
    if "candidate_error" not in error_frame.columns:
        error_frame["candidate_error"] = (
            frame["candidate_prediction"] - frame["actual"]
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=error_frame["timestamp"],
            y=error_frame["reference_error"],
            name="Reference error",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=error_frame["timestamp"],
            y=error_frame["candidate_error"],
            name="Candidate error",
            mode="lines",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=dict(
            text="Forecast error over time",
            font=dict(size=18),
            automargin=True,
        ),
        title_pad=dict(t=18, b=12),
        xaxis_title="Timestamp",
        yaxis_title="Error (kW)",
        margin=dict(t=85),
    )
    safe_plotly(fig, 400)

    with st.expander("Inspect saved prediction rows"):
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button(
            "Download selected prediction rows as CSV",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name="insightforge_selected_predictions.csv",
            mime="text/csv",
        )


# ============================================================
# PAGE 4 — MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":
    show_hero("Compare model errors using paired evaluation timestamps")

    st.markdown(
        """
        All comparisons below are computed from the saved Step 22 paired
        predictions. MAE and RMSE are error magnitudes; bias is the mean
        signed error (prediction minus actual). Lower MAE and RMSE are better.
        Bias near zero indicates less average signed error, but can conceal
        offsetting over- and under-predictions.
        """
    )

    comparison_rows = []

    for window in windows:
        part = predictions.loc[predictions["window"] == window]
        ref = metric_frame(part, "actual", "reference_prediction")
        cand = metric_frame(part, "actual", "candidate_prediction")

        comparison_rows.append(
            {
                "Window": window,
                "Hours": ref["n"],
                "Reference MAE (kW)": ref["mae"],
                "Candidate MAE (kW)": cand["mae"],
                "MAE improvement (%)": improvement_percent(
                    ref["mae"], cand["mae"]
                ),
                "Reference RMSE (kW)": ref["rmse"],
                "Candidate RMSE (kW)": cand["rmse"],
                "Reference bias (kW)": ref["bias"],
                "Candidate bias (kW)": cand["bias"],
            }
        )

    ref = metric_frame(predictions, "actual", "reference_prediction")
    cand = metric_frame(predictions, "actual", "candidate_prediction")

    comparison_rows.append(
        {
            "Window": "Pooled W2–W4",
            "Hours": ref["n"],
            "Reference MAE (kW)": ref["mae"],
            "Candidate MAE (kW)": cand["mae"],
            "MAE improvement (%)": improvement_percent(
                ref["mae"], cand["mae"]
            ),
            "Reference RMSE (kW)": ref["rmse"],
            "Candidate RMSE (kW)": cand["rmse"],
            "Reference bias (kW)": ref["bias"],
            "Candidate bias (kW)": cand["bias"],
        }
    )

    table = pd.DataFrame(comparison_rows)

    st.dataframe(
        table.style.format(
            {
                "Reference MAE (kW)": "{:.4f}",
                "Candidate MAE (kW)": "{:.4f}",
                "MAE improvement (%)": "{:+.2f}%",
                "Reference RMSE (kW)": "{:.4f}",
                "Candidate RMSE (kW)": "{:.4f}",
                "Reference bias (kW)": "{:+.4f}",
                "Candidate bias (kW)": "{:+.4f}",
            },
            na_rep="—",
        ),
        use_container_width=True,
        hide_index=True,
    )

    metric_choice = st.selectbox(
        "Metric to visualize",
        ["MAE (kW)", "RMSE (kW)", "Bias (kW)"],
    )

    if metric_choice == "MAE (kW)":
        ref_col, cand_col = "Reference MAE (kW)", "Candidate MAE (kW)"
    elif metric_choice == "RMSE (kW)":
        ref_col, cand_col = "Reference RMSE (kW)", "Candidate RMSE (kW)"
    else:
        ref_col, cand_col = "Reference bias (kW)", "Candidate bias (kW)"

    chart_data = table.loc[table["Window"] != "Pooled W2–W4"].melt(
        id_vars=["Window"],
        value_vars=[ref_col, cand_col],
        var_name="Model",
        value_name=metric_choice,
    )

    fig = px.bar(
        chart_data,
        x="Window",
        y=metric_choice,
        color="Model",
        barmode="group",
        title=f"{metric_choice} across evaluation windows",
        color_discrete_sequence=["#2874A6", "#E09F3E"],
    )

    fig.update_layout(
        title=dict(font=dict(size=18), automargin=True),
        margin=dict(t=85),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    safe_plotly(fig, 400)

    st.subheader("Frozen Step 20 holdout reference")

    st.markdown(
        """
        The earlier Step 20 report provides a broader benchmark comparison
        on its own strict holdout evaluation. These values are shown separately
        from the paired Step 22 reference metrics above.
        """
    )

    baseline = pd.DataFrame(
        [
            ["Previous day", 0.4824, 0.7294],
            ["Previous week", 0.5472, 0.7937],
            ["Hour-of-day profile", 0.5239, 0.6814],
            ["Frozen blend", 0.4873, 0.6353],
            ["Random Forest (Step 20 report)", 0.4251, 0.5786],
        ],
        columns=["Method", "MAE (kW)", "RMSE (kW)"],
    )

    st.dataframe(
        baseline.style.format(
            {"MAE (kW)": "{:.4f}", "RMSE (kW)": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Step 20's published reference metrics and Step 22's paired-cohort "
        "metrics come from their respective reports. They are intentionally "
        "not treated as the same evaluation."
    )


# ============================================================
# PAGE 5 — RELIABILITY & LIMITATIONS
# ============================================================

elif page == "Reliability & Limitations":
    show_hero("Understand where the model works and where it still struggles")

    st.subheader("Upward-consumption events")

    st.markdown(
        """
        The Step 22 protocol evaluated upward events using labels fixed before
        each evaluation window. This section compares saved predictions on
        those event timestamps only.
        """
    )

    ref_event = metric_frame(event_rows, "actual", "reference_prediction")
    cand_event = metric_frame(event_rows, "actual", "candidate_prediction")

    c1, c2, c3 = st.columns(3)
    c1.metric("Upward-event observations", f"{len(event_rows):,}")
    c2.metric("Reference event MAE", format_kw(ref_event["mae"]))
    c3.metric("Candidate event MAE", format_kw(cand_event["mae"]))

    event_compare = pd.DataFrame(
        [
            {
                "Model": "Frozen reference",
                "Event MAE (kW)": ref_event["mae"],
                "Event RMSE (kW)": ref_event["rmse"],
                "Event bias (kW)": ref_event["bias"],
            },
            {
                "Model": "Step 22 candidate",
                "Event MAE (kW)": cand_event["mae"],
                "Event RMSE (kW)": cand_event["rmse"],
                "Event bias (kW)": cand_event["bias"],
            },
        ]
    )

    st.dataframe(
        event_compare.style.format(
            {
                "Event MAE (kW)": "{:.4f}",
                "Event RMSE (kW)": "{:.4f}",
                "Event bias (kW)": "{:+.4f}",
            },
            na_rep="—",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Step 22's overall decision was FAIL. The candidate improved pooled "
        "hourly accuracy but did not satisfy all predefined upward-event "
        "criteria. It should remain an exploratory candidate, not be presented "
        "as an approved full replacement."
    )

    st.subheader("Predefined decision record")

    decisions = summary.get("decisions")
    if decisions is not None:
        if isinstance(decisions, (dict, list)):
            st.json(decisions, expanded=True)
        else:
            st.write(decisions)
    else:
        st.info("No decision details were found in the saved JSON summary.")

    st.subheader("Bootstrap uncertainty")

    bootstrap = summary.get("bootstrap")
    if bootstrap is not None:
        st.markdown(
            """
            The saved experiment used paired moving-block bootstrap intervals,
            with 12-hour, 24-hour, and 48-hour block-length analyses. The
            following section shows the saved bootstrap structure directly;
            it is not recalculated by the dashboard.
            """
        )
        st.json(bootstrap, expanded=False)
    else:
        st.info("No bootstrap details were found in the saved JSON summary.")

    with st.expander("Full saved experiment summary"):
        st.dataframe(
            pd.DataFrame(flatten_json(summary)),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Interpretation and limitations")

    st.markdown(
        """
        - **Single-household dataset:** the results do not establish
          performance for other homes, locations, or electricity systems.
        - **Chronological evaluation:** results are based on the specified
          evaluation windows and training protocol, not on a universal guarantee
          of future accuracy.
        - **Upward-event weakness:** the candidate's pooled improvements do
          not eliminate its weaker performance on rapid upward changes.
        - **Uncertainty:** bootstrap intervals describe uncertainty under the
          chosen resampling procedure; they do not resolve dataset shift or
          guarantee future performance.
        - **No production claim:** this dashboard is an analytical portfolio
          artifact, not a deployed energy-management or safety-critical system.
        """
    )

    st.subheader("Data source")

    st.markdown(
        """
        UCI Machine Learning Repository — Individual Household Electric Power
        Consumption.

        [Open the original dataset page](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

        The original archive is read directly and is not extracted or modified.
        """
    )


# ---------- Footer ----------

st.divider()
st.caption(
    "InsightForge AI · Evidence-first analytics · "
    "Historical measurements and saved chronological forecast evaluations"
)
