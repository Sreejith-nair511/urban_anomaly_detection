"""
pipeline.py
Runs the full ML pipeline once at startup and caches results in memory.
All API routes read from this singleton – no retraining per request.
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any

from .generator     import load_or_generate, generate_additional_anomalies
from .preprocessing import preprocess
from .pca_module    import apply_pca
from .clustering    import run_kmeans, run_dbscan
from .labeling      import add_labels_to_df, LABEL_NAMES
from .models        import prepare_split, train_random_forest, train_xgboost
from .evaluation    import compute_metrics


@dataclass
class PipelineState:
    df:              pd.DataFrame      = field(default=None)
    X_scaled:        np.ndarray        = field(default=None)
    X_pca:           np.ndarray        = field(default=None)
    scaler:          Any               = field(default=None)
    feature_cols:    list              = field(default=None)
    kmeans_labels:   np.ndarray        = field(default=None)
    dbscan_labels:   np.ndarray        = field(default=None)
    rf_model:        Any               = field(default=None)
    xgb_model:       Any               = field(default=None)
    rf_metrics:      dict              = field(default=None)
    xgb_metrics:     dict              = field(default=None)
    rf_importance:   list              = field(default=None)
    xgb_importance:  list              = field(default=None)
    kmeans_silhouette: float           = field(default=0.0)
    pca_variance:    list              = field(default=None)
    is_ready:        bool              = field(default=False)


# Module-level singleton
_state = PipelineState()


def get_state() -> PipelineState:
    return _state


def run_pipeline() -> PipelineState:
    global _state

    base_dir  = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    raw_path  = os.path.join(base_dir, "urban_data.csv")

    # 1. Data
    df    = load_or_generate(raw_path)
    extra = generate_additional_anomalies(n_samples=300)
    df    = pd.concat([df, extra], ignore_index=True).drop_duplicates()

    # 2. Preprocess
    df, X_scaled, scaler, feature_cols = preprocess(df)

    # 3. PCA
    X_pca, pca_model = apply_pca(X_scaled)

    # 4. Clustering
    kmeans_labels, _, km_score = run_kmeans(X_scaled)
    dbscan_labels, _           = run_dbscan(X_scaled)

    # 5. Labels
    df = add_labels_to_df(df, dbscan_labels, kmeans_labels)

    # 6. Train
    y = df["label"].values
    X_train, X_test, y_train, y_test = prepare_split(X_scaled, y)
    rf_model,  rf_imp  = train_random_forest(X_train, y_train, feature_cols)
    xgb_model, xgb_imp = train_xgboost(X_train, y_train, feature_cols)

    # 7. Evaluate
    rf_metrics  = compute_metrics(y_test, rf_model.predict(X_test))
    xgb_metrics = compute_metrics(y_test, xgb_model.predict(X_test))

    _state = PipelineState(
        df=df, X_scaled=X_scaled, X_pca=X_pca,
        scaler=scaler, feature_cols=feature_cols,
        kmeans_labels=kmeans_labels, dbscan_labels=dbscan_labels,
        rf_model=rf_model, xgb_model=xgb_model,
        rf_metrics=rf_metrics, xgb_metrics=xgb_metrics,
        rf_importance=rf_imp.to_dict(orient="records"),
        xgb_importance=xgb_imp.to_dict(orient="records"),
        kmeans_silhouette=km_score,
        pca_variance=[round(float(v), 4) for v in pca_model.explained_variance_ratio_],
        is_ready=True,
    )
    print("[pipeline] Pipeline complete and cached.")
    return _state
