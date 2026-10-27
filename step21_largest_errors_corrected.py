import csv
import io
import math
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "electricity_data.zip"

WINDOWS = [
    ("Window 1", datetime(2010, 4, 16, 22), datetime(2010, 6, 11, 21)),
    ("Window 2", datetime(2010, 6, 11, 22), datetime(2010, 8, 6, 21)),
    ("Window 3", datetime(2010, 8, 6, 22), datetime(2010, 10, 1, 21)),
    ("Window 4", datetime(2010, 10, 1, 22), datetime(2010, 11, 26, 21)),
]

PROFILE_WEIGHT = 0.80

MODEL_SETTINGS = {
    "n_estimators": 50,
    "max_depth": 12,
    "min_samples_leaf": 10,
    "max_features": 0.8,
    "n_jobs": 1,
    "random_state": 42,
}


def load_hourly_from_zip(path):
    totals = defaultdict(float)
    counts = defaultdict(int)

    with zipfile.ZipFile(path, "r") as archive:
        members = [
            name for name in archive.namelist()
            if name.lower().endswith("household_power_consumption.txt")
        ]

        if len(members) != 1:
            raise RuntimeError(
                "Expected one household_power_consumption.txt file; "
                f"found {members}"
            )

        print(f"Reading directly from ZIP: {members[0]}")
        print("The source archive will not be extracted.")

        with archive.open(members[0], "r") as binary:
            with io.TextIOWrapper(
                binary, encoding="utf-8", newline=""
            ) as stream:
                reader = csv.reader(stream, delimiter=";")
                header = next(reader)
                active_index = header.index("Global_active_power")

                for row_number, row in enumerate(reader, start=1):
                    if len(row) <= active_index:
                        continue

                    value_text = row[active_index].strip()

                    if not value_text or value_text == "?":
                        continue

                    try:
                        value = float(value_text)
                        timestamp = datetime.strptime(
                            row[0] + " " + row[1],
                            "%d/%m/%Y %H:%M:%S",
                        )
                    except (ValueError, IndexError):
                        continue

                    hour = timestamp.replace(
                        minute=0, second=0, microsecond=0
                    )

                    totals[hour] += value
                    counts[hour] += 1

                    if row_number % 500_000 == 0:
                        print(
                            f"  Source rows processed: {row_number:,}"
                        )

    hourly = {
        hour: totals[hour] / counts[hour]
        for hour in totals
        if counts[hour] >= 54
    }

    print(
        "Valid source rows aggregated into eligible hours: "
        f"{len(hourly):,}"
    )

    return hourly


def features(timestamp, hourly):
    lag24_time = timestamp - timedelta(hours=24)
    lag168_time = timestamp - timedelta(hours=168)

    if lag24_time not in hourly or lag168_time not in hourly:
        return None

    hour_angle = 2 * math.pi * timestamp.hour / 24
    weekday_angle = 2 * math.pi * timestamp.weekday() / 7

    return [
        hourly[lag24_time],
        hourly[lag168_time],
        math.sin(hour_angle),
        math.cos(hour_angle),
        math.sin(weekday_angle),
        math.cos(weekday_angle),
        float(timestamp.weekday() >= 5),
        float(timestamp.month),
    ]


def metrics(actual, predicted):
    return (
        mean_absolute_error(actual, predicted),
        math.sqrt(mean_squared_error(actual, predicted)),
    )


def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Dataset ZIP not found: {ZIP_PATH}")

    hourly = load_hourly_from_zip(ZIP_PATH)

    all_window_results = []
    pooled_actual = []
    pooled_predictions = defaultdict(list)

    holdout_actual = []
    holdout_predictions = defaultdict(list)
    holdout_records = []

    print("\nMODEL SETTINGS (fixed before evaluation)")

    for key, value in MODEL_SETTINGS.items():
        print(f"  {key}: {value}")

    print(f"  frozen profile blend weight: {PROFILE_WEIGHT:.2f}")

    for window_name, start, end in WINDOWS:
        print(f"\n{'=' * 72}")
        print(f"{window_name}: {start} through {end}")
        print("Preparing chronological training data...")

        profile_values = defaultdict(list)

        for timestamp, value in hourly.items():
            if timestamp < start:
                profile_values[timestamp.hour].append(value)

        profile = {
            hour: sum(values) / len(values)
            for hour, values in profile_values.items()
        }

        train_x = []
        train_y = []

        for timestamp, value in hourly.items():
            if timestamp >= start:
                continue

            x = features(timestamp, hourly)

            if x is not None:
                train_x.append(x)
                train_y.append(value)

        if not train_x:
            raise RuntimeError(
                f"No training rows available for {window_name}"
            )

        test_times = []
        test_x = []
        actual = []
        prev_day = []
        prev_week = []
        hour_profile = []

        for timestamp in sorted(hourly):
            if timestamp < start or timestamp > end:
                continue

            x = features(timestamp, hourly)

            if x is None or timestamp.hour not in profile:
                continue

            t24 = timestamp - timedelta(hours=24)
            t168 = timestamp - timedelta(hours=168)

            test_times.append(timestamp)
            test_x.append(x)
            actual.append(hourly[timestamp])
            prev_day.append(hourly[t24])
            prev_week.append(hourly[t168])
            hour_profile.append(profile[timestamp.hour])

        if not test_x:
            raise RuntimeError(
                f"No eligible evaluation rows for {window_name}"
            )

        print(f"Training rows: {len(train_y):,}")
        print(f"Shared evaluation hours: {len(actual):,}")
        print("Fitting one Random Forest model for this window...")

        model = RandomForestRegressor(**MODEL_SETTINGS)

        model.fit(
            np.asarray(train_x, dtype=np.float32),
            np.asarray(train_y, dtype=np.float32),
        )

        forest = model.predict(
            np.asarray(test_x, dtype=np.float32)
        )

        actual = np.asarray(actual, dtype=float)
        prev_day = np.asarray(prev_day, dtype=float)
        prev_week = np.asarray(prev_week, dtype=float)
        hour_profile = np.asarray(hour_profile, dtype=float)

        blend = (
            (1 - PROFILE_WEIGHT) * prev_day
            + PROFILE_WEIGHT * hour_profile
        )

        predictions = {
            "Previous day": prev_day,
            "Previous week": prev_week,
            "Hour-of-day profile": hour_profile,
            "Frozen blend": blend,
            "Random Forest": forest,
        }

        print("\nMAE and RMSE (kW):")

        window_result = {
            "name": window_name,
            "n": len(actual),
            "metrics": {},
        }

        for name, prediction in predictions.items():
            mae, rmse = metrics(actual, prediction)

            window_result["metrics"][name] = (mae, rmse)

            print(
                f"  {name:22s} MAE={mae:.4f}  RMSE={rmse:.4f}"
            )

            pooled_predictions[name].extend(prediction.tolist())

        pooled_actual.extend(actual.tolist())

        if window_name != "Window 1":
            holdout_actual.extend(actual.tolist())

            for name, prediction in predictions.items():
                holdout_predictions[name].extend(
                    prediction.tolist()
                )

            for i, timestamp in enumerate(test_times):
                holdout_records.append({
                    "window": window_name,
                    "timestamp": timestamp,
                    "hour": timestamp.hour,

                    "lag1": float(
                        hourly.get(
                            timestamp - timedelta(hours=1),
                            float("nan"),
                        )
                    ),
                    "lag3": float(
                        hourly.get(
                            timestamp - timedelta(hours=3),
                            float("nan"),
                        )
                    ),
                    "lag6": float(
                        hourly.get(
                            timestamp - timedelta(hours=6),
                            float("nan"),
                        )
                    ),
                    "lag24": float(prev_day[i]),
                    "lag168": float(prev_week[i]),
                    "actual": float(actual[i]),

                    **{
                        name: float(prediction[i])
                        for name, prediction in predictions.items()
                    },
                })

        all_window_results.append(window_result)

        importances = sorted(
            zip(
                [
                    "lag24",
                    "lag168",
                    "hour_sin",
                    "hour_cos",
                    "weekday_sin",
                    "weekday_cos",
                    "is_weekend",
                    "month",
                ],
                model.feature_importances_,
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        print("Top model features:")

        for feature, importance in importances[:5]:
            print(f"  {feature:14s} {importance:.4f}")

        del model, train_x, train_y, test_x, forest

    # ---------------------------------------------------------
    # POOLED RESULTS
    # ---------------------------------------------------------

    print(f"\n{'=' * 72}")
    print("POOLED RESULTS — all eligible hours across the four windows")

    actual = np.asarray(pooled_actual, dtype=float)

    print(f"Total shared evaluation hours: {len(actual):,}")

    for name, values in pooled_predictions.items():
        mae, rmse = metrics(
            actual,
            np.asarray(values, dtype=float),
        )

        print(f"  {name:22s} MAE={mae:.4f}  RMSE={rmse:.4f}")

    # ---------------------------------------------------------
    # STRICT HOLDOUT RESULTS
    # ---------------------------------------------------------

    print(f"\n{'=' * 72}")
    print("STRICT HOLDOUT RESULTS — WINDOWS 2–4 ONLY")

    holdout_y = np.asarray(holdout_actual, dtype=float)

    print(f"Total strict holdout hours: {len(holdout_y):,}")

    for name, values in holdout_predictions.items():
        mae, rmse = metrics(
            holdout_y,
            np.asarray(values, dtype=float),
        )

        print(f"  {name:22s} MAE={mae:.4f}  RMSE={rmse:.4f}")

    # ---------------------------------------------------------
    # TIME-OF-DAY ERROR ANALYSIS
    # ---------------------------------------------------------

    print(f"\n{'=' * 72}")
    print("TIME-OF-DAY ERROR ANALYSIS — STRICT HOLDOUT WINDOWS 2–4")
    print(
        "Hour | Count | RF MAE | RF RMSE | Blend MAE | Blend RMSE | "
        "Day MAE | Day RMSE"
    )

    for hour in range(24):
        rows = [
            row for row in holdout_records
            if row["hour"] == hour
        ]

        if not rows:
            continue

        y = np.asarray(
            [row["actual"] for row in rows],
            dtype=float,
        )
        rf = np.asarray(
            [row["Random Forest"] for row in rows],
            dtype=float,
        )
        blend_values = np.asarray(
            [row["Frozen blend"] for row in rows],
            dtype=float,
        )
        day = np.asarray(
            [row["Previous day"] for row in rows],
            dtype=float,
        )

        rf_mae, rf_rmse = metrics(y, rf)
        blend_mae, blend_rmse = metrics(y, blend_values)
        day_mae, day_rmse = metrics(y, day)

        print(
            f"{hour:02d}:00 | {len(rows):5d} | "
            f"{rf_mae:.4f} | {rf_rmse:.4f} | "
            f"{blend_mae:.4f} | {blend_rmse:.4f} | "
            f"{day_mae:.4f} | {day_rmse:.4f}"
        )

    # ---------------------------------------------------------
    # CONSUMPTION-LEVEL ERROR ANALYSIS
    # ---------------------------------------------------------

    print(f"\n{'=' * 72}")
    print("CONSUMPTION-LEVEL ERROR ANALYSIS — DESCRIPTIVE HOLDOUT GROUPS")

    actual_values = np.asarray(
        [row["actual"] for row in holdout_records],
        dtype=float,
    )

    low_cut, high_cut = np.quantile(
        actual_values,
        [1 / 3, 2 / 3],
    )

    print(f"Low group: actual <= {low_cut:.4f} kW")
    print(
        f"Medium group: {low_cut:.4f} < actual <= "
        f"{high_cut:.4f} kW"
    )
    print(f"High group: actual > {high_cut:.4f} kW")

    print(
        "Group  | Count | RF MAE | RF RMSE | Blend MAE | Blend RMSE | "
        "Day MAE | Day RMSE"
    )

    groups = [
        ("Low", lambda value: value <= low_cut),
        ("Medium", lambda value: low_cut < value <= high_cut),
        ("High", lambda value: value > high_cut),
    ]

    for label, predicate in groups:
        rows = [
            row for row in holdout_records
            if predicate(row["actual"])
        ]

        if not rows:
            continue

        y = np.asarray(
            [row["actual"] for row in rows],
            dtype=float,
        )
        rf = np.asarray(
            [row["Random Forest"] for row in rows],
            dtype=float,
        )
        blend_values = np.asarray(
            [row["Frozen blend"] for row in rows],
            dtype=float,
        )
        day = np.asarray(
            [row["Previous day"] for row in rows],
            dtype=float,
        )

        rf_mae, rf_rmse = metrics(y, rf)
        blend_mae, blend_rmse = metrics(y, blend_values)
        day_mae, day_rmse = metrics(y, day)

        print(
            f"{label:6s} | {len(rows):5d} | "
            f"{rf_mae:.4f} | {rf_rmse:.4f} | "
            f"{blend_mae:.4f} | {blend_rmse:.4f} | "
            f"{day_mae:.4f} | {day_rmse:.4f}"
        )

    # ---------------------------------------------------------
    # FORECAST BIAS
    # ---------------------------------------------------------

    print(f"\n{'=' * 72}")
    print("FORECAST BIAS — STRICT HOLDOUT WINDOWS 2–4")
    print("Bias = forecast minus actual; positive means overprediction.")
    print(
        "Method                 Bias (kW)   MAE (kW)   "
        "RMSE (kW)   Max |error| (kW)"
    )

    error_arrays = {}

    for name in holdout_predictions:
        predicted = np.asarray(
            holdout_predictions[name],
            dtype=float,
        )

        errors = predicted - holdout_y
        error_arrays[name] = errors

        mae, rmse = metrics(holdout_y, predicted)

        print(
            f"{name:22s} {np.mean(errors):10.4f}   "
            f"{mae:8.4f}   {rmse:9.4f}   "
            f"{np.max(np.abs(errors)):15.4f}"
        )

    # ---------------------------------------------------------
    # LARGE ABSOLUTE ERRORS
    # ---------------------------------------------------------

    print(f"\n{'=' * 72}")
    print("LARGE ABSOLUTE ERRORS — STRICT HOLDOUT WINDOWS 2–4")
    print("Method                 Threshold   Count   Percentage")

    for name, errors in error_arrays.items():
        absolute_errors = np.abs(errors)

        for threshold in (0.5, 1.0, 1.5):
            count = int(np.sum(absolute_errors > threshold))
            percentage = 100.0 * count / len(absolute_errors)

            print(
                f"{name:22s} > {threshold:3.1f} kW   "
                f"{count:5d}   {percentage:8.2f}%"
            )

    # ---------------------------------------------------------
    # EXTREME-ERROR CONTRIBUTION
    # ---------------------------------------------------------

    print(f"\n{'=' * 72}")
    print("EXTREME-ERROR CONTRIBUTION — STRICT HOLDOUT WINDOWS 2–4")
    print("Method                 Worst 5% count   Share of total squared error")

    for name, errors in error_arrays.items():
        squared_errors = np.square(errors)
        worst_count = max(
            1,
            int(np.ceil(0.05 * len(squared_errors))),
        )
        worst_indices = np.argsort(squared_errors)[-worst_count:]
        total_squared_error = float(np.sum(squared_errors))

        worst_share = (
            100.0
            * float(np.sum(squared_errors[worst_indices]))
            / total_squared_error
            if total_squared_error > 0
            else 0.0
        )

        print(
            f"{name:22s} {worst_count:14d}   "
            f"{worst_share:25.2f}%"
        )

    # =========================================================
    # STEP 21.3: LARGEST RANDOM FOREST ERRORS
    # =========================================================

    print(f"\n{'=' * 100}")
    print("STEP 21.3 — LARGEST RANDOM FOREST ERRORS")
    print("Strict holdout: Windows 2–4; model settings remain unchanged.")

    rf_errors = np.asarray(
        [
            row["Random Forest"] - row["actual"]
            for row in holdout_records
        ],
        dtype=float,
    )

    rf_abs_errors = np.abs(rf_errors)
    top_count = min(20, len(holdout_records))

    top_indices = np.argsort(rf_abs_errors)[-top_count:][::-1]

    # ---------------------------------------------------------
    # TOP 20 ERROR TABLE
    # ---------------------------------------------------------

    print(f"\nTOP {top_count} RANDOM FOREST ERRORS")

    print(
        "Rank | Window   | Timestamp           | Actual | RF forecast | "
        "RF signed error | Blend abs error | Previous-day abs error"
    )
    print("-" * 115)

    for rank, idx in enumerate(top_indices, start=1):
        row = holdout_records[int(idx)]

        rf_error = row["Random Forest"] - row["actual"]
        blend_error = abs(
            row["Frozen blend"] - row["actual"]
        )
        day_error = abs(
            row["Previous day"] - row["actual"]
        )

        print(
            f"{rank:4d} | {row['window']:8s} | "
            f"{row['timestamp']} | {row['actual']:6.3f} | "
            f"{row['Random Forest']:11.3f} | "
            f"{rf_error:+15.3f} | "
            f"{blend_error:15.3f} | "
            f"{day_error:21.3f}"
        )

    # ---------------------------------------------------------
    # CORRECTED PLACEMENT:
    # HISTORICAL CONTEXT IMMEDIATELY AFTER TOP 20 ERRORS
    # ---------------------------------------------------------

    print(f"\n{'=' * 100}")
    print("HISTORICAL CONTEXT FOR THE TOP RANDOM FOREST ERRORS")
    print("Lag values are consumption readings in kW before the forecast hour.")
    print(
        "Change24 = actual minus previous-day reading; "
        "Change168 = actual minus previous-week reading."
    )

    print(
        "Timestamp           | Actual | Lag1  | Lag3  | Lag6  | "
        "Lag24 | Lag168 | Change24 | Change168"
    )
    print("-" * 105)

    for rank, idx in enumerate(top_indices, start=1):
        row = holdout_records[int(idx)]

        print(
            f"{row['timestamp']} | {row['actual']:6.3f} | "
            f"{row['lag1']:5.3f} | "
            f"{row['lag3']:5.3f} | "
            f"{row['lag6']:5.3f} | "
            f"{row['lag24']:5.3f} | "
            f"{row['lag168']:6.3f} | "
            f"{row['actual'] - row['lag24']:+8.3f} | "
            f"{row['actual'] - row['lag168']:+9.3f}"
        )

    # ---------------------------------------------------------
    # RANDOM FOREST ERROR SUMMARY BY WINDOW
    # ---------------------------------------------------------

    print(f"\n{'=' * 100}")
    print("RANDOM FOREST ERROR SUMMARY BY WINDOW")

    print(
        "Window   | Count | MAE (kW) | RMSE (kW) | "
        "Bias (kW) | Max abs error (kW)"
    )

    for window_name in sorted(
        {row["window"] for row in holdout_records}
    ):
        rows = [
            row for row in holdout_records
            if row["window"] == window_name
        ]

        actual_values = np.asarray(
            [row["actual"] for row in rows],
            dtype=float,
        )
        predicted_values = np.asarray(
            [row["Random Forest"] for row in rows],
            dtype=float,
        )

        window_errors = predicted_values - actual_values
        window_mae, window_rmse = metrics(
            actual_values,
            predicted_values,
        )

        print(
            f"{window_name:8s} | {len(rows):5d} | "
            f"{window_mae:8.4f} | {window_rmse:9.4f} | "
            f"{np.mean(window_errors):9.4f} | "
            f"{np.max(np.abs(window_errors)):18.4f}"
        )

    # ---------------------------------------------------------
    # RANDOM FOREST ERROR SUMMARY BY HOUR
    # ---------------------------------------------------------

    print(f"\n{'=' * 100}")
    print("RANDOM FOREST ERROR SUMMARY BY HOUR OF DAY")

    print(
        "Hour | Count | MAE (kW) | RMSE (kW) | "
        "Bias (kW) | Max abs error (kW)"
    )

    for hour in range(24):
        rows = [
            row for row in holdout_records
            if row["hour"] == hour
        ]

        if not rows:
            continue

        actual_values = np.asarray(
            [row["actual"] for row in rows],
            dtype=float,
        )
        predicted_values = np.asarray(
            [row["Random Forest"] for row in rows],
            dtype=float,
        )

        hour_errors = predicted_values - actual_values
        hour_mae, hour_rmse = metrics(
            actual_values,
            predicted_values,
        )

        print(
            f"{hour:02d}:00 | {len(rows):5d} | "
            f"{hour_mae:8.4f} | {hour_rmse:9.4f} | "
            f"{np.mean(hour_errors):9.4f} | "
            f"{np.max(np.abs(hour_errors)):18.4f}"
        )

    print(
        "\nExperiment finished. Source data and previous scripts/reports "
        "were not modified."
    )


if __name__ == "__main__":
    main()