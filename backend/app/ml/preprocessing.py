import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_feature_columns() -> list:
    return ["AQI", "temperature", "humidity", "traffic_density", "noise_level", "hour", "day_of_week"]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"]   = pd.to_datetime(df["timestamp"])
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    return df

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df

def preprocess(df: pd.DataFrame) -> tuple:
    """Returns (df_enriched, X_scaled, scaler, feature_cols)."""
    feature_cols = get_feature_columns()
    df = handle_missing(df)
    df = engineer_features(df)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    return df, X_scaled, scaler, feature_cols
