"""
visualization.py
Generates and saves all plots:
  - PCA scatter with cluster coloring
  - Anomaly highlighting overlay
  - Feature importance bar charts for RF and XGB
  - Confusion matrix heatmaps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

OUTPUT_DIR = "data/processed/plots"
LABEL_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}
LABEL_NAMES  = {0: "Normal", 1: "Moderate", 2: "High Anomaly"}


def _save(fig: plt.Figure, filename: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualization] Saved → {path}")


def plot_pca_clusters(X_pca: np.ndarray, labels: np.ndarray,
                      title: str = "PCA – Cluster View") -> None:
    """PCA 2D scatter colored by label (Normal / Moderate / High Anomaly)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for lbl, color in LABEL_COLORS.items():
        mask = labels == lbl
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=color, label=LABEL_NAMES[lbl],
                   alpha=0.55, s=18, edgecolors="none")

    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(title="Label", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "pca_clusters.png")


def plot_anomaly_highlight(X_pca: np.ndarray, labels: np.ndarray) -> None:
    """Highlight anomalies (High Anomaly) against normal points in PCA space."""
    fig, ax = plt.subplots(figsize=(8, 6))

    normal_mask = labels != 2
    anomaly_mask = labels == 2

    ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1],
               c="#90CAF9", alpha=0.4, s=14, label="Normal / Moderate", edgecolors="none")
    ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1],
               c="#D32F2F", alpha=0.85, s=35, label="High Anomaly",
               marker="X", edgecolors="black", linewidths=0.3)

    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.set_title("PCA – Anomaly Highlight", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "anomaly_highlight.png")


def plot_feature_importance(importance_df: pd.DataFrame,
                             model_name: str = "Model") -> None:
    """Horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(7, 4))

    colors = sns.color_palette("Blues_r", len(importance_df))
    ax.barh(importance_df["feature"][::-1],
            importance_df["importance"][::-1],
            color=colors)

    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"{model_name} – Feature Importance", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, f"feature_importance_{model_name.lower().replace(' ', '_')}.png")


def plot_confusion_matrix(cm: np.ndarray, model_name: str = "Model") -> None:
    """Heatmap of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Normal", "Moderate", "High Anomaly"]

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax)

    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"{model_name} – Confusion Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")


def plot_label_distribution(labels: np.ndarray) -> None:
    """Bar chart showing class balance."""
    fig, ax = plt.subplots(figsize=(6, 4))
    unique, counts = np.unique(labels, return_counts=True)
    names = [LABEL_NAMES[u] for u in unique]
    colors = [LABEL_COLORS[u] for u in unique]

    ax.bar(names, counts, color=colors, edgecolor="white", linewidth=0.8)
    for i, (name, count) in enumerate(zip(names, counts)):
        ax.text(i, count + 20, str(count), ha="center", fontsize=10)

    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Label Distribution", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, "label_distribution.png")
