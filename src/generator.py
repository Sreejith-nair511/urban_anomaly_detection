"""
generator.py
Generates synthetic urban sensor data for Bangalore zones,
with injected anomalies to simulate real-world irregularities.
"""

import numpy as np
import pandas as pd
import os


BANGALORE_ZONES = [
    "Koramangala", "Whitefield", "Indiranagar",
    "Hebbal", "Electronic_City", "Jayanagar",
    "Marathahalli", "Yelahanka", "BTM_Layout", "HSR_Layout"
]


def generate_base_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate normal urban sensor readings."""
    np.random.seed(seed)

    timestamps = pd.date_range(start="2024-01-01", periods=n_samples, freq="30min")
    location_ids = np.random.choice(BANGALORE_ZONES, size=n_samples)

    data = pd.DataFrame({
        "timestamp": timestamps,
        "location_id": location_ids,
        "AQI": np.random.uniform(50, 200, n_samples),
        "temperature": np.random.uniform(18, 35, n_samples),
        "humidity": np.random.uniform(35, 85, n_samples),
        "traffic_density": np.random.uniform(10, 70, n_samples),
        "noise_level": np.random.uniform(35, 90, n_samples),
    })
    return data


def inject_anomalies(df: pd.DataFrame, anomaly_fraction: float = 0.13) -> pd.DataFrame:
    """
    Inject realistic anomalies into the dataset.
    Anomaly types: AQI spikes, traffic surges, noise bursts, combined events.
    """
    df = df.copy()
    n = len(df)
    n_anomalies = int(n * anomaly_fraction)

    anomaly_indices = np.random.choice(df.index, size=n_anomalies, replace=False)

    # Split anomaly indices into three groups
    split1 = n_anomalies // 3
    split2 = 2 * n_anomalies // 3

    # AQI pollution spikes
    aqi_idx = anomaly_indices[:split1]
    df.loc[aqi_idx, "AQI"] = np.random.uniform(310, 500, len(aqi_idx))
    df.loc[aqi_idx, "humidity"] = np.random.uniform(75, 90, len(aqi_idx))

    # Traffic surges
    traffic_idx = anomaly_indices[split1:split2]
    df.loc[traffic_idx, "traffic_density"] = np.random.uniform(82, 100, len(traffic_idx))
    df.loc[traffic_idx, "noise_level"] = np.random.uniform(85, 120, len(traffic_idx))

    # Combined environmental anomalies
    combined_idx = anomaly_indices[split2:]
    df.loc[combined_idx, "AQI"] = np.random.uniform(280, 450, len(combined_idx))
    df.loc[combined_idx, "traffic_density"] = np.random.uniform(78, 100, len(combined_idx))
    df.loc[combined_idx, "noise_level"] = np.random.uniform(82, 115, len(combined_idx))
    df.loc[combined_idx, "temperature"] = np.random.uniform(36, 40, len(combined_idx))

    df["is_anomaly"] = 0
    df.loc[anomaly_indices, "is_anomaly"] = 1

    return df


def generate_additional_anomalies(n_samples: int = 300, seed: int = 99) -> pd.DataFrame:
    """
    Generate additional extreme anomaly samples for augmentation.
    Criteria: AQI > 300, traffic_density > 80, noise_level > 80.
    """
    np.random.seed(seed)

    timestamps = pd.date_range(start="2024-06-01", periods=n_samples, freq="1h")
    location_ids = np.random.choice(BANGALORE_ZONES, size=n_samples)

    extra = pd.DataFrame({
        "timestamp": timestamps,
        "location_id": location_ids,
        "AQI": np.random.uniform(301, 500, n_samples),
        "temperature": np.random.uniform(30, 40, n_samples),
        "humidity": np.random.uniform(60, 90, n_samples),
        "traffic_density": np.random.uniform(81, 100, n_samples),
        "noise_level": np.random.uniform(81, 120, n_samples),
        "is_anomaly": 1,
    })
    return extra


def save_raw_data(df: pd.DataFrame, path: str = "data/raw/urban_data.csv") -> None:
    """Save the generated dataset to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[generator] Saved {len(df)} records to {path}")


def load_or_generate(path: str = "data/raw/urban_data.csv") -> pd.DataFrame:
    """Load existing data or generate fresh if not found."""
    if os.path.exists(path):
        print(f"[generator] Loading existing data from {path}")
        return pd.read_csv(path, parse_dates=["timestamp"])

    print("[generator] Generating synthetic dataset...")
    base = generate_base_data()
    base = inject_anomalies(base)
    extra = generate_additional_anomalies()
    df = pd.concat([base, extra], ignore_index=True)
    save_raw_data(df, path)
    return df
