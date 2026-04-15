"use client";
import { useEffect, useState } from "react";
import { fetchMetrics, fetchSummary, Metrics, DataSummary } from "../lib/api";
import { KpiCard, Card, SectionTitle, Spinner, ErrorMsg } from "../components/Card";
import { AqiHistogram, ZoneAnomalyBar, HourlyTrend } from "../components/Charts";
import { Database, AlertTriangle, TrendingUp, Activity, Cpu } from "lucide-react";

export default function OverviewPage() {
  const [metrics, setMetrics]   = useState<Metrics | null>(null);
  const [summary, setSummary]   = useState<DataSummary | null>(null);
  const [loading, setLoading]   = useState(true);
  const [error,   setError]     = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchMetrics(), fetchSummary()])
      .then(([m, s]) => { setMetrics(m.data); setSummary(s.data); })
      .catch(() => setError("Failed to load data. Is the backend running on port 8000?"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Spinner />;
  if (error)   return <ErrorMsg msg={error} />;
  if (!metrics || !summary) return null;

  const dist = summary.label_distribution;
  const totalAnomalies = (dist["High Anomaly"] ?? 0) + (dist["Moderate"] ?? 0);

  return (
    <div className="space-y-8 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-white">Overview</h1>
        <p className="text-sm text-slate-500 mt-1">
          Real-time ML pipeline monitoring Bangalore's urban environment
        </p>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        <KpiCard
          icon={<Database size={18} />}
          label="Total Samples"
          value={summary.total_samples.toLocaleString()}
          sub="Synthetic + augmented"
          color="cyan"
        />
        <KpiCard
          icon={<AlertTriangle size={18} />}
          label="Anomalies Detected"
          value={totalAnomalies.toLocaleString()}
          sub={`${((totalAnomalies / summary.total_samples) * 100).toFixed(1)}% of dataset`}
          color="red"
        />
        <KpiCard
          icon={<TrendingUp size={18} />}
          label="RF Accuracy"
          value={`${(metrics.random_forest.accuracy * 100).toFixed(1)}%`}
          sub={`F1: ${metrics.random_forest.f1_score}`}
          color="emerald"
        />
        <KpiCard
          icon={<Cpu size={18} />}
          label="XGB Accuracy"
          value={`${(metrics.xgboost.accuracy * 100).toFixed(1)}%`}
          sub={`F1: ${metrics.xgboost.f1_score}`}
          color="indigo"
        />
      </div>

      {/* Model metrics row */}
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        {[
          { label: "RF Precision",  value: metrics.random_forest.precision, color: "text-cyan-400" },
          { label: "RF Recall",     value: metrics.random_forest.recall,    color: "text-cyan-400" },
          { label: "XGB Precision", value: metrics.xgboost.precision,       color: "text-indigo-400" },
          { label: "XGB Recall",    value: metrics.xgboost.recall,          color: "text-indigo-400" },
        ].map((m) => (
          <Card key={m.label} className="flex items-center justify-between">
            <span className="text-xs text-slate-500">{m.label}</span>
            <span className={`text-lg font-bold ${m.color}`}>{m.value}</span>
          </Card>
        ))}
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <Card>
          <SectionTitle>AQI Distribution</SectionTitle>
          <AqiHistogram data={summary.aqi_histogram} />
        </Card>
        <Card>
          <SectionTitle>Anomaly Rate by Zone</SectionTitle>
          <ZoneAnomalyBar data={summary.zone_anomaly_rates} />
        </Card>
      </div>

      {/* Label distribution */}
      <Card>
        <SectionTitle>Label Distribution</SectionTitle>
        <div className="grid grid-cols-3 gap-4">
          {Object.entries(dist).map(([label, count]) => {
            const pct = ((count / summary.total_samples) * 100).toFixed(1);
            const color = label === "Normal" ? "emerald" : label === "Moderate" ? "amber" : "red";
            const barColor = label === "Normal" ? "bg-emerald-500" : label === "Moderate" ? "bg-amber-500" : "bg-red-500";
            return (
              <div key={label} className={`rounded-xl border p-4 bg-${color}-500/5 border-${color}-500/20`}>
                <p className="text-xs text-slate-400 mb-1">{label}</p>
                <p className="text-2xl font-bold text-white">{count.toLocaleString()}</p>
                <div className="mt-2 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div className={`h-full ${barColor} rounded-full`} style={{ width: `${pct}%` }} />
                </div>
                <p className="text-xs text-slate-500 mt-1">{pct}%</p>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}
