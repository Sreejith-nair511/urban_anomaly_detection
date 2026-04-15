"""
labeling.py
Derives final anomaly labels from DBSCAN and K-Means outputs.

Label rules:
  - DBSCAN == -1           → "High Anomaly"  (2)
  - K-Means cluster == 1   → "Moderate"      (1)
  - Otherwise              → "Normal"         (0)
"""

import numpy as np
import pandas as pd


LABEL_MAP = {
    "Normal": 0,
    "Moderate": 1,
    "High Anomaly": 2,
}

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


def assign_labels(dbscan_labels: np.ndarray, kmeans_labels: np.ndarray) -> np.ndarray:
    """
    Apply label engineering rules to produce a numeric target array.

    Priority: DBSCAN anomaly > K-Means moderate > Normal
    """
    n = len(dbscan_labels)
    labels = np.zeros(n, dtype=int)  # default: Normal

    # K-Means cluster 1 → Moderate
    labels[kmeans_labels == 1] = LABEL_MAP["Moderate"]

    # DBSCAN noise → High Anomaly (overrides Moderate)
    labels[dbscan_labels == -1] = LABEL_MAP["High Anomaly"]

    counts = dict(zip(*np.unique(labels, return_counts=True)))
    readable = {LABEL_NAMES[k]: v for k, v in counts.items()}
    print(f"[labeling] Label distribution: {readable}")

    return labels


def add_labels_to_df(df: pd.DataFrame,
                     dbscan_labels: np.ndarray,
                     kmeans_labels: np.ndarray) -> pd.DataFrame:
    """Attach cluster labels and final encoded label to the dataframe."""
    df = df.copy()
    df["kmeans_cluster"] = kmeans_labels
    df["dbscan_cluster"] = dbscan_labels
    df["label"] = assign_labels(dbscan_labels, kmeans_labels)
    df["label_name"] = df["label"].map(LABEL_NAMES)
    return df
