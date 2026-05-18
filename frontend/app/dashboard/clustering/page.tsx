"use client";
import { useEffect, useState } from "react";
import { fetchSummary, DataSummary } from "../../lib/api";
import { Card, SectionTitle, Spinner, ErrorMsg } from "../../components/Card";
import { PcaScatter, ZoneAnomalyBar } from "../../components/Charts";
import { ScatterChart, Filter } from "lucide-react";
import clsx from "clsx";

type LabelFilter = "All" | "Normal" | "Moderate" | "High Anomaly";

const LABEL_CFG = {
  All:           { color: "text-slate-400",   bg: "bg-slate-500/10",   border: "border-slate-500/30"   },
  Normal:        { color: "text-green-400",   bg: "bg-green-500/10",   border: "border-green-500/30"   },
  Moderate:      { color: "text-amber-400",   bg: "bg-amber-500/10",   border: "border-amber-500/30"   },
  "High Anomaly":{ color: "text-red-400",     bg: "bg-red-500/10",     border: "border-red-500/30"     },
};

export default function ClusteringPage() {
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);
  const [filter, setFilter]   = useState<LabelFilter>("All");

  useEffect(() => {
    fetchSummary()
      .then(r => setSummary(r.data))
      .catch(() => setError("Backend offline. Start FastAPI on port 8000."))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Spinner />;
  if (error)   return <ErrorMsg msg={error} />;
  if (!summary) return null;

  const filtered = filter === "All"
    ? summary.pca_scatter
    : summary.pca_scatter.filter(d => d.label === filter);

  const counts = {
    Normal:         summary.pca_scatter.filter(d => d.label === "Normal").length,
    Moderate:       summary.pca_scatter.filter(d => d.label === "Moderate").length,
    "High Anomaly": summary.pca_scatter.filter(d => d.label === "High Anomaly").length,
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-purple-500 to-violet-700 flex items-center justify-center">
            <ScatterChart size={16} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl font-black text-white" style={{ fontFamily:"'Orbitron',sans-serif" }}>AI CLUSTERING</h1>
            <p className="text-[10px] text-slate-500 font-mono">PCA 2D projection · DBSCAN density analysis · K-Means zones</p>
          </div>
        </div>
      </div>

      {/* Cluster count stats */}
      <div className="grid grid-cols-3 gap-3">
        {(["Normal","Moderate","High Anomaly"] as const).map(l => {
          const cfg = LABEL_CFG[l];
          const icon = l==="Normal" ? "🟢" : l==="Moderate" ? "🟡" : "🔴";
          return (
            <div key={l} className={clsx("glass-card rounded-xl p-4 border", cfg.border, cfg.bg)}>
              <div className="flex items-center gap-2 mb-1">
                <span>{icon}</span>
                <p className={clsx("text-[9px] font-bold font-mono tracking-wider", cfg.color)}>{l.toUpperCase()}</p>
              </div>
              <p className={clsx("text-2xl font-black font-mono", cfg.color)} style={{ fontFamily:"'Orbitron',sans-serif" }}>
                {counts[l].toLocaleString()}
              </p>
              <p className="text-[9px] text-slate-600 font-mono mt-1">sample points</p>
            </div>
          );
        })}
      </div>

      {/* Filter row */}
      <div className="flex items-center gap-2">
        <Filter size={12} className="text-slate-600" />
        <span className="text-[9px] text-slate-600 font-mono uppercase tracking-widest mr-1">Filter:</span>
        {(["All","Normal","Moderate","High Anomaly"] as LabelFilter[]).map(f => {
          const cfg = LABEL_CFG[f];
          return (
            <button key={f} onClick={() => setFilter(f)}
              className={clsx("text-[9px] font-bold font-mono px-3 py-1.5 rounded-lg border transition-all",
                filter === f ? clsx(cfg.color, cfg.border, cfg.bg) : "text-slate-600 border-slate-700/40 hover:text-slate-300"
              )}>
              {f === "All" ? "ALL POINTS" : f.toUpperCase()}
              <span className="ml-1.5 opacity-60">
                {f === "All" ? summary.pca_scatter.length : counts[f as keyof typeof counts]}
              </span>
            </button>
          );
        })}
      </div>

      {/* PCA Scatter */}
      <Card glow="purple">
        <SectionTitle>PCA 2D CLUSTER VISUALIZATION — {filter.toUpperCase()}</SectionTitle>
        <PcaScatter data={filtered} />
        <div className="mt-4 grid grid-cols-3 gap-3 text-[9px] font-mono">
          <div className="p-3 bg-green-500/5 border border-green-500/20 rounded-xl">
            <p className="text-green-400 font-bold">🟢 NORMAL CLUSTER</p>
            <p className="text-slate-600 mt-1">Dense core region. Low DBSCAN neighbor distance. K-Means centroid proximity high.</p>
          </div>
          <div className="p-3 bg-amber-500/5 border border-amber-500/20 rounded-xl">
            <p className="text-amber-400 font-bold">🟡 MODERATE CLUSTER</p>
            <p className="text-slate-600 mt-1">K-Means Cluster 1. Boundary region — elevated but not extreme parameter combinations.</p>
          </div>
          <div className="p-3 bg-red-500/5 border border-red-500/20 rounded-xl">
            <p className="text-red-400 font-bold">🔴 HIGH ANOMALY</p>
            <p className="text-slate-600 mt-1">DBSCAN noise points (label −1). Low-density isolated outliers in 7D sensor space.</p>
          </div>
        </div>
      </Card>

      {/* Algorithm explanations */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Card glow="red">
          <SectionTitle>DBSCAN — DENSITY-BASED ANOMALY DETECTION</SectionTitle>
          <div className="space-y-2.5">
            {[
              { step:"ε = 1.5",        desc:"Epsilon radius — the neighborhood distance in normalized 7D space"    },
              { step:"MinPts = 15",    desc:"Minimum neighbors required to qualify as a core point"               },
              { step:"Core Points",   desc:"Dense region members — classified as Normal or Moderate by K-Means"    },
              { step:"Noise (−1)",    desc:"Isolated outliers with < 15 neighbors — labeled High Anomaly directly" },
            ].map(s => (
              <div key={s.step} className="flex items-start gap-3 p-2.5 rounded-lg bg-red-500/5 border border-red-500/10">
                <span className="text-[9px] font-black text-red-400 font-mono w-20 flex-shrink-0 mt-0.5">{s.step}</span>
                <p className="text-[9px] text-slate-500 font-mono">{s.desc}</p>
              </div>
            ))}
          </div>
        </Card>

        <Card glow="cyan">
          <SectionTitle>K-MEANS — CENTROID PARTITIONING</SectionTitle>
          <div className="space-y-2.5">
            {[
              { step:"k = 3",          desc:"Three clusters: Normal baseline, Moderate activity, High stress"     },
              { step:"n_init = 10",    desc:"10 random initializations — best inertia selected for stability"     },
              { step:"Cluster 0",      desc:"Largest, tightest cluster — Normal urban baseline conditions"        },
              { step:"Cluster 1",      desc:"Boundary cluster — elevated parameters → Moderate Anomaly label"     },
            ].map(s => (
              <div key={s.step} className="flex items-start gap-3 p-2.5 rounded-lg bg-cyan-500/5 border border-cyan-500/10">
                <span className="text-[9px] font-black text-cyan-400 font-mono w-20 flex-shrink-0 mt-0.5">{s.step}</span>
                <p className="text-[9px] text-slate-500 font-mono">{s.desc}</p>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Zone anomaly rates */}
      <Card glow="amber">
        <SectionTitle>ZONE ANOMALY RATES — ALL LOCATIONS</SectionTitle>
        <ZoneAnomalyBar data={summary.zone_anomaly_rates} />
      </Card>
    </div>
  );
}
