import numpy as np
from sklearn.decomposition import PCA

def apply_pca(X_scaled: np.ndarray, n_components: int = 2) -> tuple:
    pca   = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca
