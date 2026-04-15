"use client";
import { useEffect, useState } from "react";
import { fetchSummary, fetchMetrics, DataSummary, Metrics } from "../../lib/api";
import { Card, SectionTitle, Spinner, ErrorMsg } from "../../components/Card";
import { AqiHistogram, HourlyTrend, FeatureImportanceBar } from "../../components/Charts";

export default function AnalyticsPage() {
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchSummary(), fetchMetrics()])
      .then(([s, m]) => { setSummary(s.data); setMetrics(m.data); })
      .catch(() => setError("Failed to load analytics data."))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Spinner />;
  if (error)   return <ErrorMsg msg={error} />;
  if (!summary || !metrics) return null;

  return (
    <div className="space-y-8 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-white">Analytics</h1>
        <p className="text-sm text-slate-500 mt-1">Feature distributions, trends, and model insights</p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <Card>
          <SectionTitle>AQI Distribution</SectionTitle>
          <AqiHistogram data={summary.aqi_histogram} />
          <p className="text-xs text-slate-600 mt-2">
            Green = Good (50–200) · Amber = Moderate (200–350) · Red = Hazardous (350+)
          </p>
        </Card>

        <Card>
          <SectionTitle>Hourly Sensor Trends</SectionTitle>
          <HourlyTrend data={summary.hourly_trends} />
          <p className="text-xs text-slate-600 mt-2">
            Average readings across all zones by hour of day
          </p>
        </Card>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <Card>
          <SectionTitle>Random Forest – Feature Importance</SectionTitle>
          <FeatureImportanceBar data={metrics.rf_importance} color="#06b6d4" />
        </Card>
        <Card>
          <SectionTitle>XGBoost – Feature Importance</SectionTitle>
          <FeatureImportanceBar data={metrics.xgb_importance} color="#818cf8" />
        </Card>
      </div>

      {/* PCA variance */}
      <Card>
        <SectionTitle>PCA Explained Variance</SectionTitle>
        <div className="grid grid-cols-2 gap-6">
          {metrics.pca_variance.map((v, i) => (
            <div key={i}>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-400">PC{i + 1}</span>
                <span className="text-cyan-400 font-semibold">{(v * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-cyan-500 to-indigo-500 rounded-full"
                  style={{ width: `${v * 100}%` }}
                />
              </div>
            </div>
          ))}
          <div className="col-span-2 text-xs text-slate-600 mt-1">
            Total variance explained: {(metrics.pca_variance.reduce((a, b) => a + b, 0) * 100).toFixed(1)}%
          </div>
        </div>
      </Card>

      {/* Silhouette score */}
      <Card>
        <SectionTitle>Clustering Quality</SectionTitle>
        <div className="flex items-center gap-6">
          <div>
            <p className="text-xs text-slate-500 mb-1">K-Means Silhouette Score</p>
            <p className="text-3xl font-bold text-cyan-400">{metrics.kmeans_silhouette.toFixed(4)}</p>
            <p className="text-xs text-slate-600 mt-1">Range: -1 (bad) → +1 (perfect)</p>
          </div>
          <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-cyan-500 to-emerald-500 rounded-full"
              style={{ width: `${((metrics.kmeans_silhouette + 1) / 2) * 100}%` }}
            />
          </div>
        </div>
      </Card>
    </div>
  );
}
