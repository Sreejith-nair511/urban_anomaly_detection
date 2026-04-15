"""
data_loader.py
Handles loading data from disk and basic validation.
"""

import pandas as pd
import os


REQUIRED_COLUMNS = [
    "timestamp", "location_id", "AQI",
    "temperature", "humidity", "traffic_density", "noise_level"
]


def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV data from the given path.
    Validates required columns and parses timestamps.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[data_loader] File not found: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[data_loader] Missing columns: {missing}")

    print(f"[data_loader] Loaded {len(df)} rows, {len(df.columns)} columns from {path}")
    return df


def get_feature_columns() -> list:
    """Return the list of numeric feature columns used for ML."""
    return [
        "AQI", "temperature", "humidity",
        "traffic_density", "noise_level",
        "hour", "day_of_week"
    ]
