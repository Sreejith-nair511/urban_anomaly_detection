from fastapi import APIRouter, HTTPException
import numpy as np

from ..schemas.input_schema import PredictInput
from ..ml.pipeline import get_state
from ..ml.labeling import LABEL_NAMES

router = APIRouter()


@router.post("/predict")
def predict(body: PredictInput):
    state = get_state()
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")

    # Build feature vector in the same order as training
    feat_map = {
        "AQI":             body.AQI,
        "temperature":     body.temperature,
        "humidity":        body.humidity,
        "traffic_density": body.traffic_density,
        "noise_level":     body.noise_level,
        "hour":            body.hour,
        "day_of_week":     body.day_of_week,
    }
    X = np.array([[feat_map[c] for c in state.feature_cols]])
    X_scaled = state.scaler.transform(X)

    rf_pred   = int(state.rf_model.predict(X_scaled)[0])
    xgb_pred  = int(state.xgb_model.predict(X_scaled)[0])
    rf_proba  = state.rf_model.predict_proba(X_scaled)[0].tolist()
    xgb_proba = state.xgb_model.predict_proba(X_scaled)[0].tolist()

    return {
        "rf_prediction":   LABEL_NAMES[rf_pred],
        "xgb_prediction":  LABEL_NAMES[xgb_pred],
        "rf_confidence":   {LABEL_NAMES[i]: round(p, 4) for i, p in enumerate(rf_proba)},
        "xgb_confidence":  {LABEL_NAMES[i]: round(p, 4) for i, p in enumerate(xgb_proba)},
    }


@router.get("/metrics")
def metrics():
    state = get_state()
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")
    return {
        "random_forest": state.rf_metrics,
        "xgboost":       state.xgb_metrics,
        "kmeans_silhouette": state.kmeans_silhouette,
        "pca_variance":  state.pca_variance,
        "rf_importance": state.rf_importance,
        "xgb_importance": state.xgb_importance,
    }


@router.get("/data-summary")
def data_summary():
    state = get_state()
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")

    df = state.df
    label_dist = df["label_name"].value_counts().to_dict()

    # Zone anomaly rates
    zone_anomaly = (
        df.groupby("location_id")
          .apply(lambda x: round(float((x["label"] == 2).mean() * 100), 2))
          .reset_index(name="anomaly_rate")
          .rename(columns={"location_id": "zone"})
          .to_dict(orient="records")
    )

    # Hourly averages for trend chart
    hourly = (
        df.groupby("hour")[["AQI", "traffic_density", "noise_level"]]
          .mean().round(2).reset_index()
          .to_dict(orient="records")
    )

    # PCA scatter sample (max 1500 points for perf)
    sample = df.sample(min(1500, len(df)), random_state=42)
    idx    = sample.index
    pca_scatter = [
        {
            "x":     round(float(state.X_pca[i, 0]), 4),
            "y":     round(float(state.X_pca[i, 1]), 4),
            "label": df.loc[i, "label_name"],
            "zone":  df.loc[i, "location_id"],
        }
        for i in idx
    ]

    # AQI histogram buckets
    bins   = list(range(50, 510, 50))
    labels = [f"{b}-{b+50}" for b in bins[:-1]]
    counts, _ = np.histogram(df["AQI"].values, bins=bins)
    aqi_hist  = [{"range": l, "count": int(c)} for l, c in zip(labels, counts)]

    return {
        "total_samples":  int(len(df)),
        "label_distribution": label_dist,
        "zone_anomaly_rates": zone_anomaly,
        "hourly_trends":  hourly,
        "pca_scatter":    pca_scatter,
        "aqi_histogram":  aqi_hist,
    }
