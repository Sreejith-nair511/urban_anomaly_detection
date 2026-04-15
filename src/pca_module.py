"""
pca_module.py
Applies PCA for dimensionality reduction to 2 components.
Used for visualization and as input to clustering.
"""

import numpy as np
from sklearn.decomposition import PCA


def apply_pca(X_scaled: np.ndarray, n_components: int = 2) -> tuple:
    """
    Fit and transform data using PCA.

    Returns:
        X_pca: 2D transformed array
        pca: fitted PCA object (for explained variance access)
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    total = explained.sum() * 100
    print(f"[pca] Explained variance per component: {[f'{v:.3f}' for v in explained]}")
    print(f"[pca] Total variance explained: {total:.2f}%")

    return X_pca, pca
