import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def run_kmeans(X: np.ndarray, n_clusters: int = 3) -> tuple:
    model  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    score  = silhouette_score(X, labels)
    return labels, model, float(score)

def run_dbscan(X: np.ndarray, eps: float = 1.5, min_samples: int = 15) -> tuple:
    model  = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels, model
