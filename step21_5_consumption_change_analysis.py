
import csv
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

import step21_largest_errors_corrected as baseline


ROOT = Path(__file__).resolve().parent

RECORDS_PATH = ROOT / "STEP21_5_CONSUMPTION_CHANGE_RECORDS.csv"
SUMMARY_PATH = ROOT / "STEP21_5_CONSUMPTION_CHANGE_SUMMARY.csv"

CHANGE_QUANTILE = 0.90
LARGE_ERROR_THRESHOLD_KW = 1.0


def hourly_change(timestamp, hourly):
    """Return the change from the immediately preceding eligible hour."""
    previous_time = timestamp - baseline.timedelta(hours=1)

    if previous_time not in hourly:
        return None

    return hourly[timestamp] - hourly[previous_time]


def error_metrics(records):
    """Summarize forecast errors for a group of evaluation records."""
    if not records:
        return {
            "count": 0,
            "mae_kW": "",
            "rmse_kW": "",
            "bias_kW": "",
            "errors_over_1kW_pct": "",
        }

    errors = np.asarray(
        [row["forecast_error_kW"] for row in records],
        dtype=float,
    )

    return {
        "count": len(errors),
        "mae_kW": float(np.mean(np.abs(errors))),
        "rmse_kW": float(np.sqrt(np.mean(errors ** 2))),
        "bias_kW": float(np.mean(errors)),
        "errors_over_1kW_pct": float(
            100 * np.mean(np.abs(errors) > LARGE_ERROR_THRESHOLD_KW)
        ),
    }


def main():
    hourly = baseline.load_hourly_from_zip(baseline.ZIP_PATH)

    all_holdout = []
    window_thresholds = {}

    print("\nSTEP 21.5 — FULL-HOLDOUT CONSUMPTION-CHANGE ANALYSIS")
    print("Frozen Random Forest settings; no model tuning.")
    print(f"Change threshold: training-data {CHANGE_QUANTILE:.0%} quantile")
    print(f"Large forecast error: absolute error > {LARGE_ERROR_THRESHOLD_KW:.1f} kW")
    print("Window 1 is used for historical development, not pooled holdout scoring.")

    for window_name, start, end in baseline.WINDOWS:
        profile_values = {}

        for timestamp, value in hourly.items():
            if timestamp < start:
                profile_values.setdefault(timestamp.hour, []).append(value)

        profile = {
            hour: float(np.mean(values))
            for hour, values in profile_values.items()
        }

        train_x, train_y = [], []

        for timestamp, value in sorted(hourly.items()):
            if timestamp >= start:
                continue

            features = baseline.features(timestamp, hourly)

            if features is not None:
                train_x.append(features)
                train_y.append(value)

        # Determine the event threshold from past observations only.
        historical_changes = []

        for timestamp in sorted(hourly):
            if timestamp >= start:
                break

            change = hourly_change(timestamp, hourly)

            if change is not None:
                historical_changes.append(abs(change))

        if not historical_changes:
            raise RuntimeError(
                f"No historical hourly changes available for {window_name}"
            )

        threshold = float(
            np.quantile(historical_changes, CHANGE_QUANTILE)
        )
        window_thresholds[window_name] = threshold

        test_times, test_x, actual = [], [], []

        for timestamp in sorted(hourly):
            if timestamp < start or timestamp > end:
                continue

            features = baseline.features(timestamp, hourly)

            if features is None or timestamp.hour not in profile:
                continue

            test_times.append(timestamp)
            test_x.append(features)
            actual.append(hourly[timestamp])

        if not train_x or not test_x:
            raise RuntimeError(
                f"Insufficient training/evaluation data for {window_name}"
            )

        model = RandomForestRegressor(**baseline.MODEL_SETTINGS)
        model.fit(
            np.asarray(train_x, dtype=np.float32),
            np.asarray(train_y, dtype=np.float32),
        )

        predictions = model.predict(
            np.asarray(test_x, dtype=np.float32)
        )

        print(
            f"\n{window_name}: training rows={len(train_y):,}; "
            f"evaluation hours={len(test_times):,}; "
            f"frozen change threshold={threshold:.4f} kW"
        )

        # The first window is not part of the strict holdout.
        if window_name == "Window 1":
            del model
            continue

        for timestamp, observed, forecast in zip(
            test_times, actual, predictions
        ):
            change = hourly_change(timestamp, hourly)

            if change is None:
                category = "unclassifiable_missing_previous_hour"
            elif change >= threshold:
                category = "upward_event"
            elif change <= -threshold:
                category = "downward_event"
            else:
                category = "ordinary_change"

            error = float(forecast - observed)

            all_holdout.append({
                "window": window_name,
                "timestamp": timestamp.isoformat(sep=" "),
                "actual_kW": float(observed),
                "rf_forecast_kW": float(forecast),
                "forecast_error_kW": error,
                "absolute_error_kW": abs(error),
                "hourly_change_kW": (
                    float(change) if change is not None else ""
                ),
                "change_threshold_kW": threshold,
                "change_category": category,
                "large_error_over_1kW": abs(error) > LARGE_ERROR_THRESHOLD_KW,
            })

        del model

    categories = [
        "upward_event",
        "downward_event",
        "ordinary_change",
        "unclassifiable_missing_previous_hour",
    ]

    summary_rows = []

    for window_name in ["ALL_HOLDOUT", "Window 2", "Window 3", "Window 4"]:
        window_records = [
            row for row in all_holdout
            if window_name == "ALL_HOLDOUT" or row["window"] == window_name
        ]

        for category in categories:
            group = [
                row for row in window_records
                if row["change_category"] == category
            ]

            result = error_metrics(group)

            thresholds = (
                [row["change_threshold_kW"] for row in window_records]
                if window_records else []
            )

            summary_rows.append({
                "scope": window_name,
                "change_category": category,
                "threshold_method": "90th percentile of absolute hourly changes in pre-window history",
                "threshold_kW": (
                    float(np.mean(thresholds)) if thresholds else ""
                ),
                **result,
            })

    record_fields = [
        "window",
        "timestamp",
        "actual_kW",
        "rf_forecast_kW",
        "forecast_error_kW",
        "absolute_error_kW",
        "hourly_change_kW",
        "change_threshold_kW",
        "change_category",
        "large_error_over_1kW",
    ]

    with RECORDS_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=record_fields)
        writer.writeheader()
        writer.writerows(all_holdout)

    summary_fields = [
        "scope",
        "change_category",
        "threshold_method",
        "threshold_kW",
        "count",
        "mae_kW",
        "rmse_kW",
        "bias_kW",
        "errors_over_1kW_pct",
    ]

    with SUMMARY_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nTotal strict-holdout records: {len(all_holdout):,}")
    print("\nSUMMARY — MAE, RMSE, bias and large-error frequency")

    for row in summary_rows:
        if row["scope"] != "ALL_HOLDOUT":
            continue

        print(
            f"{row['change_category']}: "
            f"n={row['count']}, "
            f"MAE={row['mae_kW']}, "
            f"RMSE={row['rmse_kW']}, "
            f"bias={row['bias_kW']}, "
            f"errors >1 kW (%)={row['errors_over_1kW_pct']}"
        )

    print(f"\nDetailed records saved: {RECORDS_PATH.name}")
    print(f"Summary saved: {SUMMARY_PATH.name}")
    print("Original dataset and earlier reports were not modified.")


if __name__ == "__main__":
    main()
