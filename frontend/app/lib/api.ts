import axios from "axios";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const api = axios.create({ baseURL: BASE, timeout: 30000 });

// ── Types ──────────────────────────────────────────────────────────────────

export interface PredictInput {
  AQI: number;
  temperature: number;
  humidity: number;
  traffic_density: number;
  noise_level: number;
  hour: number;
  day_of_week: number;
}

export interface PredictResult {
  rf_prediction:  string;
  xgb_prediction: string;
  rf_confidence:  Record<string, number>;
  xgb_confidence: Record<string, number>;
}

export interface Metrics {
  random_forest:      { accuracy: number; precision: number; recall: number; f1_score: number };
  xgboost:            { accuracy: number; precision: number; recall: number; f1_score: number };
  kmeans_silhouette:  number;
  pca_variance:       number[];
  rf_importance:      { feature: string; importance: number }[];
  xgb_importance:     { feature: string; importance: number }[];
}

export interface DataSummary {
  total_samples:      number;
  label_distribution: Record<string, number>;
  zone_anomaly_rates: { zone: string; anomaly_rate: number }[];
  hourly_trends:      { hour: number; AQI: number; traffic_density: number; noise_level: number }[];
  pca_scatter:        { x: number; y: number; label: string; zone: string }[];
  aqi_histogram:      { range: string; count: number }[];
}

// ── API calls ──────────────────────────────────────────────────────────────

export const fetchHealth   = ()                    => api.get("/health");
export const fetchMetrics  = ()                    => api.get<Metrics>("/api/metrics");
export const fetchSummary  = ()                    => api.get<DataSummary>("/api/data-summary");
export const postPredict   = (body: PredictInput)  => api.post<PredictResult>("/api/predict", body);
