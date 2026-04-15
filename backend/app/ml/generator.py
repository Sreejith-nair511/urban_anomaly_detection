import numpy as np
import pandas as pd
import os

BANGALORE_ZONES = [
    "Koramangala", "Whitefield", "Indiranagar", "Hebbal",
    "Electronic_City", "Jayanagar", "Marathahalli",
    "Yelahanka", "BTM_Layout", "HSR_Layout",
]

def generate_base_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    timestamps   = pd.date_range(start="2024-01-01", periods=n_samples, freq="30min")
    location_ids = np.random.choice(BANGALORE_ZONES, size=n_samples)
    return pd.DataFrame({
        "timestamp":       timestamps,
        "location_id":     location_ids,
        "AQI":             np.random.uniform(50,  200, n_samples),
        "temperature":     np.random.uniform(18,  35,  n_samples),
        "humidity":        np.random.uniform(35,  85,  n_samples),
        "traffic_density": np.random.uniform(10,  70,  n_samples),
        "noise_level":     np.random.uniform(35,  90,  n_samples),
    })

def inject_anomalies(df: pd.DataFrame, anomaly_fraction: float = 0.13) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    n_anomalies = int(n * anomaly_fraction)
    idx = np.random.choice(df.index, size=n_anomalies, replace=False)
    s1, s2 = n_anomalies // 3, 2 * n_anomalies // 3

    df.loc[idx[:s1],  "AQI"]             = np.random.uniform(310, 500, s1)
    df.loc[idx[:s1],  "humidity"]         = np.random.uniform(75,  90,  s1)
    df.loc[idx[s1:s2],"traffic_density"]  = np.random.uniform(82,  100, s2 - s1)
    df.loc[idx[s1:s2],"noise_level"]      = np.random.uniform(85,  120, s2 - s1)
    df.loc[idx[s2:],  "AQI"]             = np.random.uniform(280, 450, n_anomalies - s2)
    df.loc[idx[s2:],  "traffic_density"]  = np.random.uniform(78,  100, n_anomalies - s2)
    df.loc[idx[s2:],  "noise_level"]      = np.random.uniform(82,  115, n_anomalies - s2)
    df.loc[idx[s2:],  "temperature"]      = np.random.uniform(36,  40,  n_anomalies - s2)

    df["is_anomaly"] = 0
    df.loc[idx, "is_anomaly"] = 1
    return df

def generate_additional_anomalies(n_samples: int = 300, seed: int = 99) -> pd.DataFrame:
    np.random.seed(seed)
    timestamps   = pd.date_range(start="2024-06-01", periods=n_samples, freq="1h")
    location_ids = np.random.choice(BANGALORE_ZONES, size=n_samples)
    return pd.DataFrame({
        "timestamp":       timestamps,
        "location_id":     location_ids,
        "AQI":             np.random.uniform(301, 500, n_samples),
        "temperature":     np.random.uniform(30,  40,  n_samples),
        "humidity":        np.random.uniform(60,  90,  n_samples),
        "traffic_density": np.random.uniform(81,  100, n_samples),
        "noise_level":     np.random.uniform(81,  120, n_samples),
        "is_anomaly":      1,
    })

def load_or_generate(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["timestamp"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base  = inject_anomalies(generate_base_data())
    extra = generate_additional_anomalies()
    df    = pd.concat([base, extra], ignore_index=True)
    df.to_csv(path, index=False)
    return df
