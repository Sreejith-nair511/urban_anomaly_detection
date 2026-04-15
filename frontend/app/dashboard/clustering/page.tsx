"use client";
import { useEffect, useState } from "react";
import { fetchSummary, DataSummary } from "../../lib/api";
import { Card, SectionTitle, Spinner, ErrorMsg } from "../../components/Card";
import { PcaScatter, ZoneAnomalyBar } from "../../components/Charts";

export default function ClusteringPage() {
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState<string | null>(null);
  const [filter,  setFilter]  = useState<string>("All");

  useEffect(() => {
    fetchSummary()
      .then((r) => setSummary(r.data))
      .catch(() => setError("Failed to load clustering data."))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Spinner />;
  if (error)   return <ErrorMsg msg={error} />;
  if (!summary) return null;

  const labels = ["All", "Normal", "Moderate", "High Anomaly"];
  const filtered = filter === "All"
    ? summary.pca_scatter
    : summary.pca_scatter.filter((d) => d.label === filter);

  const anomalyCount  = summary.pca_scatter.filter((d) => d.label === "High Anomaly").length;
  const moderateCount = summary.pca_scatter.filter((d) => d.label === "Moderate").length;
  const normalCount   = summary.pca_scatter.filter((d) => d.label === "Normal").length;

  return (
    <div className="space-y-8 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-white">Clustering Analysis</h1>
        <p className="text-sm text-slate-500 mt-1">
          PCA 2D projection with K-Means + DBSCAN labels (sample of 1,500 points)
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: "Normal",       count: normalCount,   color: "emerald" },
          { label: "Moderate",     count: moderateCount, color: "amber"   },
          { label: "High Anomaly", count: anomalyCount,  color: "red"     },
        ].map(({ label, count, color }) => (
          <Card key={label} className={`border-${color}-500/20 bg-${color}-500/5`}>
            <p className="text-xs text-slate-400">{label}</p>
            <p className="text-2xl font-bold text-white mt-1">{count.toLocaleString()}</p>
            <p className="text-xs text-slate-500 mt-0.5">in sample</p>
          </Card>
        ))}
      </div>

      {/* Filter */}
      <div className="flex gap-2">
        {labels.map((l) => (
          <button
            key={l}
            onClick={() => setFilter(l)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              filter === l
                ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30"
                : "bg-slate-800 text-slate-400 border border-slate-700 hover:border-slate-600"
            }`}
          >
            {l}
          </button>
        ))}
      </div>

      {/* PCA scatter */}
      <Card>
        <SectionTitle>PCA Scatter Plot – {filter}</SectionTitle>
        <PcaScatter data={filtered} />
        <div className="flex gap-4 mt-3">
          {[
            { label: "Normal",       color: "bg-emerald-500" },
            { label: "Moderate",     color: "bg-amber-500"   },
            { label: "High Anomaly", color: "bg-red-500"     },
          ].map(({ label, color }) => (
            <div key={label} className="flex items-center gap-1.5 text-xs text-slate-400">
              <span className={`w-2.5 h-2.5 rounded-full ${color}`} />
              {label}
            </div>
          ))}
        </div>
      </Card>

      {/* Zone anomaly */}
      <Card>
        <SectionTitle>Anomaly Rate by Bangalore Zone</SectionTitle>
        <ZoneAnomalyBar data={summary.zone_anomaly_rates} />
      </Card>
    </div>
  );
}
