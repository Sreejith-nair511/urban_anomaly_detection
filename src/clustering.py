"""
clustering.py
Implements K-Means and DBSCAN clustering.
DBSCAN label=-1 indicates detected anomalies.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


def run_kmeans(X: np.ndarray, n_clusters: int = 3, seed: int = 42) -> tuple:
    """
    Fit K-Means clustering.

    Returns:
        labels: cluster assignments per sample
        model: fitted KMeans object
    """
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)
    print(f"[clustering] K-Means silhouette score: {score:.4f}")
    print(f"[clustering] K-Means cluster distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    return labels, model


def run_dbscan(X: np.ndarray, eps: float = 1.5, min_samples: int = 15) -> tuple:
    """
    Fit DBSCAN clustering.
    Points labeled -1 are considered anomalies (noise points).

    eps=1.5, min_samples=15 tuned for normalized 7-feature urban data
    to yield ~10-20% anomaly rate.

    Returns:
        labels: cluster assignments (-1 = anomaly)
        model: fitted DBSCAN object
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    n_anomalies = np.sum(labels == -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    anomaly_pct = n_anomalies / len(labels) * 100

    print(f"[clustering] DBSCAN clusters found: {n_clusters}")
    print(f"[clustering] DBSCAN anomalies detected: {n_anomalies} ({anomaly_pct:.1f}%)")

    # Silhouette only valid when more than 1 cluster and no all-noise
    valid_mask = labels != -1
    if len(set(labels[valid_mask])) > 1:
        score = silhouette_score(X[valid_mask], labels[valid_mask])
        print(f"[clustering] DBSCAN silhouette score (non-noise): {score:.4f}")

    return labels, model
