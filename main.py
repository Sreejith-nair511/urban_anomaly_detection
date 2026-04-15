"""
main.py
Smart Urban Anomaly Detection System – Bangalore
Full end-to-end ML pipeline orchestrator.
"""

import os
import sys

# Ensure src/ is importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generator      import load_or_generate, generate_additional_anomalies
from data_loader    import load_data, get_feature_columns
from preprocessing  import preprocess
from pca_module     import apply_pca
from clustering     import run_kmeans, run_dbscan
from labeling       import add_labels_to_df
from models         import prepare_train_test, train_random_forest, train_xgboost, predict
from evaluation     import evaluate_model
from visualization  import (
    plot_pca_clusters,
    plot_anomaly_highlight,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_label_distribution,
)

import pandas as pd

RAW_PATH       = "data/raw/urban_data.csv"
PROCESSED_PATH = "data/processed/urban_data_labeled.csv"


def main():
    print("\n" + "="*55)
    print("  Smart Urban Anomaly Detection System – Bangalore")
    print("="*55 + "\n")

    # ── 1. Load / Generate Data ──────────────────────────────
    print("── Step 1: Data Generation / Loading ──")
    df = load_or_generate(RAW_PATH)

    # Augment with additional extreme anomaly samples
    extra = generate_additional_anomalies(n_samples=300)
    df = pd.concat([df, extra], ignore_index=True).drop_duplicates()
    print(f"[main] Total records after augmentation: {len(df)}")

    # ── 2. Preprocessing ─────────────────────────────────────
    print("\n── Step 2: Preprocessing ──")
    feature_cols = get_feature_columns()
    df, X_scaled, scaler = preprocess(df, feature_cols)

    # ── 3. PCA ───────────────────────────────────────────────
    print("\n── Step 3: PCA ──")
    X_pca, pca_model = apply_pca(X_scaled, n_components=2)

    # ── 4. Clustering ────────────────────────────────────────
    print("\n── Step 4: Clustering ──")
    kmeans_labels, kmeans_model = run_kmeans(X_scaled, n_clusters=3)
    dbscan_labels, dbscan_model = run_dbscan(X_scaled, eps=1.5, min_samples=15)

    # ── 5. Label Engineering ─────────────────────────────────
    print("\n── Step 5: Label Engineering ──")
    df = add_labels_to_df(df, dbscan_labels, kmeans_labels)

    # Save labeled dataset
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"[main] Labeled dataset saved to {PROCESSED_PATH}")

    # ── 6. Supervised Model Training ─────────────────────────
    print("\n── Step 6: Model Training ──")
    y = df["label"].values
    X_train, X_test, y_train, y_test = prepare_train_test(X_scaled, y)

    rf_model,  rf_importance  = train_random_forest(X_train, y_train, feature_cols)
    xgb_model, xgb_importance = train_xgboost(X_train, y_train, feature_cols)

    # ── 7. Evaluation ────────────────────────────────────────
    print("\n── Step 7: Evaluation ──")
    rf_preds  = predict(rf_model,  X_test)
    xgb_preds = predict(xgb_model, X_test)

    rf_metrics  = evaluate_model(y_test, rf_preds,  "Random Forest")
    xgb_metrics = evaluate_model(y_test, xgb_preds, "XGBoost")

    # ── 8. Visualization ─────────────────────────────────────
    print("\n── Step 8: Visualization ──")
    labels = df["label"].values

    plot_pca_clusters(X_pca, labels)
    plot_anomaly_highlight(X_pca, labels)
    plot_feature_importance(rf_importance,  "Random Forest")
    plot_feature_importance(xgb_importance, "XGBoost")
    plot_confusion_matrix(rf_metrics["confusion_matrix"],  "Random Forest")
    plot_confusion_matrix(xgb_metrics["confusion_matrix"], "XGBoost")
    plot_label_distribution(labels)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "="*55)
    print("  Pipeline Complete")
    print(f"  RF  Accuracy : {rf_metrics['accuracy']:.4f}")
    print(f"  XGB Accuracy : {xgb_metrics['accuracy']:.4f}")
    print(f"  Plots saved  : data/processed/plots/")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
