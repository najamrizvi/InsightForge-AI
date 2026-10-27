import csv
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

import step21_largest_errors_corrected as baseline

ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "STEP21_4_TEMPORAL_EVENT_RESULTS.csv"
TOP_COUNT = 20
CONTEXT_HOURS = 6


def main():
    hourly = baseline.load_hourly_from_zip(baseline.ZIP_PATH)
    holdout = []

    print("\nSTEP 21.4 — TEMPORAL-EVENT DIAGNOSTIC")
    print("Using the existing frozen model settings.")
    print("No tuning or model selection will be performed.")

    for window_name, start, end in baseline.WINDOWS:
        profile_values = defaultdict(list)

        for timestamp, value in hourly.items():
            if timestamp < start:
                profile_values[timestamp.hour].append(value)

        profile = {
            hour: sum(values) / len(values)
            for hour, values in profile_values.items()
        }

        train_x, train_y = [], []

        for timestamp, value in sorted(hourly.items()):
            if timestamp >= start:
                continue
            x = baseline.features(timestamp, hourly)
            if x is not None:
                train_x.append(x)
                train_y.append(value)

        test_times, test_x, actual = [], [], []

        for timestamp in sorted(hourly):
            if timestamp < start or timestamp > end:
                continue
            x = baseline.features(timestamp, hourly)
            if x is None or timestamp.hour not in profile:
                continue
            test_times.append(timestamp)
            test_x.append(x)
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
            f"{window_name}: {len(train_y):,} training rows; "
            f"{len(test_times):,} eligible evaluation hours"
        )

        if window_name != "Window 1":
            for timestamp, observed, forecast in zip(
                test_times, actual, predictions
            ):
                holdout.append({
                    "window": window_name,
                    "timestamp": timestamp,
                    "actual": float(observed),
                    "forecast": float(forecast),
                    "error": float(forecast - observed),
                    "abs_error": float(abs(forecast - observed)),
                })

        del model

    holdout.sort(key=lambda row: row["abs_error"], reverse=True)
    top_events = holdout[:TOP_COUNT]

    fields = [
        "event_rank", "window", "event_timestamp", "context_timestamp",
        "hours_from_event", "actual_kW", "rf_forecast_kW",
        "forecast_error_kW", "absolute_error_kW",
        "change_from_previous_hour_kW",
    ]

    with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        for rank, event in enumerate(top_events, start=1):
            event_time = event["timestamp"]
            print(
                f"\n#{rank:02d} {event['window']} | {event_time} | "
                f"actual={event['actual']:.3f} kW | "
                f"forecast={event['forecast']:.3f} kW | "
                f"error={event['error']:+.3f} kW"
            )
            print("  Nearby eligible hourly readings:")

            for offset in range(-CONTEXT_HOURS, CONTEXT_HOURS + 1):
                timestamp = event_time + timedelta(hours=offset)
                observed = hourly.get(timestamp)

                if observed is None:
                    print(
                        f"    {timestamp} | unavailable/ineligible hour"
                    )
                    writer.writerow({
                        "event_rank": rank,
                        "window": event["window"],
                        "event_timestamp": event_time.isoformat(sep=" "),
                        "context_timestamp": timestamp.isoformat(sep=" "),
                        "hours_from_event": offset,
                        "actual_kW": "",
                        "rf_forecast_kW": "",
                        "forecast_error_kW": "",
                        "absolute_error_kW": "",
                        "change_from_previous_hour_kW": "",
                    })
                    continue

                previous = hourly.get(timestamp - timedelta(hours=1))
                change = observed - previous if previous is not None else None

                print(
                    f"    {timestamp} | actual={observed:.3f} kW | "
                    f"hourly change="
                    f"{change:+.3f} kW" if change is not None
                    else f"    {timestamp} | actual={observed:.3f} kW | "
                         "hourly change unavailable"
                )

                writer.writerow({
                    "event_rank": rank,
                    "window": event["window"],
                    "event_timestamp": event_time.isoformat(sep=" "),
                    "context_timestamp": timestamp.isoformat(sep=" "),
                    "hours_from_event": offset,
                    "actual_kW": f"{observed:.6f}",
                    "rf_forecast_kW": (
                        f"{event['forecast']:.6f}" if offset == 0 else ""
                    ),
                    "forecast_error_kW": (
                        f"{event['error']:.6f}" if offset == 0 else ""
                    ),
                    "absolute_error_kW": (
                        f"{event['abs_error']:.6f}" if offset == 0 else ""
                    ),
                    "change_from_previous_hour_kW": (
                        f"{change:.6f}" if change is not None else ""
                    ),
                })

    print(f"\nDiagnostic CSV saved to: {REPORT_PATH.name}")
    print("Original scripts, dataset ZIP, and previous reports were not modified.")


if __name__ == "__main__":
    main()
