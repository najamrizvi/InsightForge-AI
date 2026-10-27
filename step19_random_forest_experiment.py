import csv
import io
import math
import zipfile
from collections import defaultdict
from datetime import datetime
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
                f"Expected one household_power_consumption.txt file; found {members}"
            )

        print(f"Reading directly from ZIP: {members[0]}")
        print("The source archive will not be extracted.")

        with archive.open(members[0], "r") as binary:
            with io.TextIOWrapper(binary, encoding="utf-8", newline="") as stream:
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
                            row[0] + " " + row[1], "%d/%m/%Y %H:%M:%S"
                        )
                    except (ValueError, IndexError):
                        continue

                    hour = timestamp.replace(minute=0, second=0, microsecond=0)
                    totals[hour] += value
                    counts[hour] += 1

                    if row_number % 500_000 == 0:
                        print(f"  Source rows processed: {row_number:,}")

    hourly = {
        hour: totals[hour] / counts[hour]
        for hour in totals
        if counts[hour] >= 54
    }
    print(f"Valid source rows aggregated into eligible hours: {len(hourly):,}")
    return hourly

def features(timestamp, hourly):
    lag24_time = timestamp.replace() - __import__("datetime").timedelta(hours=24)
    lag168_time = timestamp.replace() - __import__("datetime").timedelta(hours=168)

    if lag24_time not in hourly or lag168_time not in hourly:
        return None

    hour_angle = 2 * math.pi * timestamp.hour / 24
    weekday_angle = 2 * math.pi * timestamp.weekday() / 7

    x = [
        hourly[lag24_time],
        hourly[lag168_time],
        math.sin(hour_angle),
        math.cos(hour_angle),
        math.sin(weekday_angle),
        math.cos(weekday_angle),
        float(timestamp.weekday() >= 5),
        float(timestamp.month),
    ]
    return x

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
            raise RuntimeError(f"No training rows available for {window_name}")

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

            t24 = timestamp - __import__("datetime").timedelta(hours=24)
            t168 = timestamp - __import__("datetime").timedelta(hours=168)

            test_times.append(timestamp)
            test_x.append(x)
            actual.append(hourly[timestamp])
            prev_day.append(hourly[t24])
            prev_week.append(hourly[t168])
            hour_profile.append(profile[timestamp.hour])

        if not test_x:
            raise RuntimeError(f"No eligible evaluation rows for {window_name}")

        print(f"Training rows: {len(train_y):,}")
        print(f"Shared evaluation hours: {len(actual):,}")
        print("Fitting one Random Forest model for this window...")

        model = RandomForestRegressor(**MODEL_SETTINGS)
        model.fit(np.asarray(train_x, dtype=np.float32),
                  np.asarray(train_y, dtype=np.float32))
        forest = model.predict(np.asarray(test_x, dtype=np.float32))

        actual = np.asarray(actual, dtype=float)
        prev_day = np.asarray(prev_day, dtype=float)
        prev_week = np.asarray(prev_week, dtype=float)
        hour_profile = np.asarray(hour_profile, dtype=float)
        blend = (1 - PROFILE_WEIGHT) * prev_day + PROFILE_WEIGHT * hour_profile

        predictions = {
            "Previous day": prev_day,
            "Previous week": prev_week,
            "Hour-of-day profile": hour_profile,
            "Frozen blend": blend,
            "Random Forest": forest,
        }

        print("\nMAE and RMSE (kW):")
        window_result = {"name": window_name, "n": len(actual), "metrics": {}}
        for name, prediction in predictions.items():
            mae, rmse = metrics(actual, prediction)
            window_result["metrics"][name] = (mae, rmse)
            print(f"  {name:22s} MAE={mae:.4f}  RMSE={rmse:.4f}")
            pooled_predictions[name].extend(prediction.tolist())

        pooled_actual.extend(actual.tolist())
        all_window_results.append(window_result)

        importances = sorted(
            zip(
                ["lag24", "lag168", "hour_sin", "hour_cos",
                 "weekday_sin", "weekday_cos", "is_weekend", "month"],
                model.feature_importances_,
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        print("Top model features:")
        for feature, importance in importances[:5]:
            print(f"  {feature:14s} {importance:.4f}")

        del model, train_x, train_y, test_x, forest

    print(f"\n{'=' * 72}")
    print("POOLED RESULTS — all eligible hours across the four windows")
    actual = np.asarray(pooled_actual, dtype=float)
    print(f"Total shared evaluation hours: {len(actual):,}")
    for name, values in pooled_predictions.items():
        mae, rmse = metrics(actual, np.asarray(values, dtype=float))
        print(f"  {name:22s} MAE={mae:.4f}  RMSE={rmse:.4f}")

    print("\nExperiment finished. No source data or Step 18 checkpoint was modified.")

if __name__ == "__main__":
    main()

