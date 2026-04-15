"""
evaluation.py
Computes classification metrics and confusion matrix for model evaluation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   model_name: str = "Model") -> dict:
    """
    Compute accuracy, precision, recall, F1, and confusion matrix.

    Returns a dict of all metrics.
    """
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    print(f"\n{'='*45}")
    print(f"  {model_name} Evaluation")
    print(f"{'='*45}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1 Score : {metrics['f1_score']:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_true, y_pred,
                                 target_names=["Normal", "Moderate", "High Anomaly"],
                                 zero_division=0))
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}\n")

    return metrics
