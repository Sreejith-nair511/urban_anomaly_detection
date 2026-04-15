"""
models.py
Trains RandomForestClassifier and XGBClassifier on labeled urban data.
Returns predictions and feature importances for both models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def prepare_train_test(X: np.ndarray, y: np.ndarray,
                        test_size: float = 0.2,
                        seed: int = 42) -> tuple:
    """Split data into 80/20 train/test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    print(f"[models] Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                         feature_names: list) -> tuple:
    """
    Train a RandomForestClassifier.

    Returns:
        model: fitted RF model
        feature_importance_df: sorted DataFrame of feature importances
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("[models] Random Forest trained.")
    print(importance_df.to_string(index=False))
    return model, importance_df


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                   feature_names: list) -> tuple:
    """
    Train an XGBClassifier.

    Returns:
        model: fitted XGB model
        feature_importance_df: sorted DataFrame of feature importances
    """
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("[models] XGBoost trained.")
    print(importance_df.to_string(index=False))
    return model, importance_df


def predict(model, X_test: np.ndarray) -> np.ndarray:
    """Return predictions from a trained model."""
    return model.predict(X_test)
