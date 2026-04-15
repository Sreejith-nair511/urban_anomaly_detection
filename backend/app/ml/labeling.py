import numpy as np
import pandas as pd

LABEL_NAMES = {0: "Normal", 1: "Moderate", 2: "High Anomaly"}

def assign_labels(dbscan_labels: np.ndarray, kmeans_labels: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(dbscan_labels), dtype=int)
    labels[kmeans_labels == 1] = 1   # Moderate
    labels[dbscan_labels == -1] = 2  # High Anomaly
    return labels

def add_labels_to_df(df: pd.DataFrame, dbscan_labels, kmeans_labels) -> pd.DataFrame:
    df = df.copy()
    df["kmeans_cluster"] = kmeans_labels
    df["dbscan_cluster"]  = dbscan_labels
    df["label"]           = assign_labels(dbscan_labels, kmeans_labels)
    df["label_name"]      = df["label"].map(LABEL_NAMES)
    return df
