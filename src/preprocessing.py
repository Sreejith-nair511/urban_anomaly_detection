"""
preprocessing.py
Handles missing value imputation, feature engineering,
and StandardScaler normalization.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with column medians."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[preprocessing] Filled NaNs in '{col}' with median={median_val:.2f}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from timestamp:
    - hour: captures intra-day patterns (rush hours, night lows)
    - day_of_week: captures weekly cycles
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    return df


def normalize_features(df: pd.DataFrame, feature_cols: list,
                        scaler_path: str = "data/processed/scaler.pkl") -> tuple:
    """
    Fit StandardScaler on feature columns and return scaled array + scaler.
    Saves scaler to disk for reproducibility.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[preprocessing] Scaler saved to {scaler_path}")

    return X_scaled, scaler


def preprocess(df: pd.DataFrame, feature_cols: list) -> tuple:
    """Full preprocessing pipeline: clean → engineer → scale."""
    df = handle_missing_values(df)
    df = engineer_features(df)
    X_scaled, scaler = normalize_features(df, feature_cols)
    print(f"[preprocessing] Preprocessing complete. Shape: {X_scaled.shape}")
    return df, X_scaled, scaler
