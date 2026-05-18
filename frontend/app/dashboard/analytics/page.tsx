"use client";
import { useEffect, useState } from "react";
import { fetchMetrics, fetchSummary, Metrics, DataSummary } from "../../lib/api";
import { Card, SectionTitle, Spinner, ErrorMsg } from "../../components/Card";
import { HourlyTrend, FeatureImportanceBar, AqiHistogram } from "../../components/Charts";
import { BarChart3 } from "lucide-react";
import clsx from "clsx";

export default function AnalyticsPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);
  const [activeModel, setActiveModel] = useState<"rf"|"xgb">("rf");

  useEffect(() => {
    Promise.all([fetchMetrics(), fetchSummary()])
      .then(([m, s]) => { setMetrics(m.data); setSummary(s.data); })
      .catch(() => setError("Backend offline. Start FastAPI on port 8000."))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Spinner />;
  if (error)   return <ErrorMsg msg={error} />;
  if (!metrics || !summary) return null;

  const imp  = activeModel === "rf" ? metrics.rf_importance : metrics.xgb_importance;
  const rf   = metrics.random_forest;
  const xgb  = metrics.xgboost;
  const pc1  = (metrics.pca_variance[0] * 100).toFixed(1);
  const pc2  = (metrics.pca_variance[1] * 100).toFixed(1);
  const cumV = ((metrics.pca_variance[0] + metrics.pca_variance[1]) * 100).toFixed(1);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-700 flex items-center justify-center">
          <BarChart3 size={16} className="text-white" />
        </div>
        <div>
          <h1 className="text-xl font-black text-white" style={{ fontFamily:"'Orbitron',sans-serif" }}>ANALYTICS</h1>
          <p className="text-[10px] text-slate-500 font-mono">Model metrics · Feature intelligence · Temporal sensor analysis</p>
        </div>
      </div>

      {/* Model score cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label:"RF Accuracy",   val:`${(rf.accuracy*100).toFixed(2)}%`,  border:"border-cyan-500/20"   },
          { label:"RF F1-Score",   val:String(rf.f1_score),                  border:"border-cyan-500/20"   },
          { label:"XGB Accuracy",  val:`${(xgb.accuracy*100).toFixed(2)}%`, border:"border-indigo-500/20" },
          { label:"XGB F1-Score",  val:String(xgb.f1_score),                border:"border-indigo-500/20" },
        ].map(m => (
          <div key={m.label} className={`glass-card rounded-xl p-3 border ${m.border}`}>
            <p className="text-[9px] text-slate-600 font-mono uppercase tracking-wider mb-1">{m.label}</p>
            <p className="text-xl font-black font-mono text-white">{m.val}</p>
          </div>
        ))}
      </div>

      {/* RF + XGB detail */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {[
          { key:"rf",  title:"🌲 RANDOM FOREST",  cfg:rf,  color:"cyan",   hint:"Bagging ensemble – 200 parallel trees vote on anomaly class." },
          { key:"xgb", title:"⚡ XGBOOST",        cfg:xgb, color:"indigo", hint:"Boosting ensemble – sequential residual correction for high precision." },
        ].map(({ key, title, cfg, color, hint }) => (
          <Card key={key} glow={color as any}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <span className={`w-0.5 h-4 bg-${color}-400 rounded-full`} />
                <p className="text-[10px] font-bold text-slate-300 font-mono tracking-widest">{title}</p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3 mb-3">
              {[
                { label:"Accuracy",  val: key==="rf" ? `${(rf.accuracy*100).toFixed(2)}%` : `${(xgb.accuracy*100).toFixed(2)}%`  },
                { label:"F1 Score",  val: key==="rf" ? String(rf.f1_score)  : String(xgb.f1_score)  },
                { label:"Precision", val: key==="rf" ? String(rf.precision) : String(xgb.precision) },
                { label:"Recall",    val: key==="rf" ? String(rf.recall)    : String(xgb.recall)    },
              ].map(m => (
                <div key={m.label} className="bg-slate-800/30 rounded-xl p-3 border border-slate-700/30">
                  <p className="text-[9px] text-slate-600 font-mono uppercase tracking-wider mb-1">{m.label}</p>
                  <p className={`text-xl font-black font-mono text-${color}-400`}>{m.val}</p>
                </div>
              ))}
            </div>
            <div className={`p-2.5 bg-${color}-500/5 border border-${color}-500/15 rounded-xl`}>
              <p className="text-[9px] text-slate-500 font-mono">⟶ {hint}</p>
            </div>
          </Card>
        ))}
      </div>

      {/* Feature importance */}
      <Card glow="cyan">
        <div className="flex items-center justify-between mb-4">
          <SectionTitle>FEATURE IMPORTANCE ANALYSIS</SectionTitle>
          <div className="flex gap-2">
            {(["rf","xgb"] as const).map(m => (
              <button key={m} onClick={() => setActiveModel(m)}
                className={clsx("text-[9px] font-bold font-mono px-3 py-1.5 rounded-lg border transition-all",
                  activeModel === m
                    ? m==="rf" ? "bg-cyan-500/20 text-cyan-400 border-cyan-500/40" : "bg-indigo-500/20 text-indigo-400 border-indigo-500/40"
                    : "bg-slate-800/40 text-slate-500 border-slate-700/40 hover:text-slate-300"
                )}>
                {m==="rf" ? "🌲 RF" : "⚡ XGB"}
              </button>
            ))}
          </div>
        </div>
        <FeatureImportanceBar data={imp} color={activeModel==="rf" ? "#06b6d4" : "#818cf8"} />
        <div className="mt-4 grid grid-cols-3 gap-2">
          {imp.slice(0,3).map((f,i) => (
            <div key={f.feature} className="bg-slate-800/30 rounded-lg p-2.5 border border-slate-700/30 text-center">
              <p className="text-[8px] text-slate-600 font-mono">#{i+1} DRIVER</p>
              <p className="text-[10px] text-slate-300 font-bold font-mono mt-0.5">{f.feature}</p>
              <p className={`text-[10px] font-mono ${activeModel==="rf"?"text-cyan-400":"text-indigo-400"}`}>{(f.importance*100).toFixed(1)}%</p>
            </div>
          ))}
        </div>
      </Card>

      {/* PCA + Silhouette */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Card glow="purple">
          <SectionTitle>PCA DIMENSIONALITY REDUCTION</SectionTitle>
          <div className="space-y-4">
            {[{ label:"PC1 — Primary",   val:pc1, color:"from-purple-500 to-indigo-500", glow:"rgba(139,92,246,0.6)" },
              { label:"PC2 — Secondary", val:pc2, color:"from-blue-500 to-cyan-500",     glow:"rgba(59,130,246,0.6)"  }].map(p => (
              <div key={p.label}>
                <div className="flex justify-between text-[9px] font-mono mb-1.5">
                  <span className="text-slate-500">{p.label}</span>
                  <span className="text-purple-400 font-bold">{p.val}%</span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div className={`h-full bg-gradient-to-r ${p.color} rounded-full`}
                    style={{ width:`${p.val}%`, boxShadow:`0 0 8px ${p.glow}`, transition:"width 1s ease" }} />
                </div>
              </div>
            ))}
            <div className="p-3 bg-purple-500/5 border border-purple-500/20 rounded-xl">
              <p className="text-[10px] text-purple-400 font-bold font-mono">{cumV}% TOTAL VARIANCE CAPTURED</p>
              <p className="text-[9px] text-slate-600 font-mono mt-1">7-dimensional sensor space compressed to 2D for visualization</p>
            </div>
          </div>
        </Card>

        <Card glow="emerald">
          <SectionTitle>K-MEANS SILHOUETTE SCORE</SectionTitle>
          <div className="flex flex-col items-center py-4">
            <div className="relative w-28 h-28">
              <svg className="w-full h-full -rotate-90" viewBox="0 0 80 80">
                <circle cx="40" cy="40" r="34" fill="none" stroke="rgba(30,41,59,0.8)" strokeWidth="6"/>
                <circle cx="40" cy="40" r="34" fill="none" stroke="#10b981" strokeWidth="6"
                  strokeDasharray={`${metrics.kmeans_silhouette * 213.6} 213.6`} strokeLinecap="round"
                  style={{ filter:"drop-shadow(0 0 8px rgba(16,185,129,0.6))" }}/>
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <p className="text-2xl font-black text-emerald-400 font-mono">{metrics.kmeans_silhouette.toFixed(3)}</p>
                <p className="text-[9px] text-slate-600 font-mono">/ 1.000</p>
              </div>
            </div>
            <p className="text-[9px] text-emerald-400 font-mono font-bold mt-2">WELL-SEPARATED CLUSTERS</p>
          </div>
          <div className="p-3 bg-emerald-500/5 border border-emerald-500/15 rounded-xl">
            <p className="text-[9px] text-slate-500 font-mono">
              ⟶ Score {metrics.kmeans_silhouette.toFixed(3)} indicates clear spatial separation between Normal,
              Moderate, and High Anomaly clusters in the 7D feature space.
            </p>
          </div>
        </Card>
      </div>

      {/* Hourly Trends */}
      <Card glow="cyan">
        <SectionTitle>24H DIURNAL SENSOR TRENDS</SectionTitle>
        <HourlyTrend data={summary.hourly_trends} />
      </Card>

      {/* AQI histogram */}
      <Card glow="amber">
        <SectionTitle>AQI DISTRIBUTION HISTOGRAM</SectionTitle>
        <AqiHistogram data={summary.aqi_histogram} />
        <div className="mt-3 p-3 bg-amber-500/5 border border-amber-500/15 rounded-xl">
          <p className="text-[9px] text-slate-500 font-mono">
            ⟶ Bimodal distribution confirms the synthetic generation model: normal urban baseline (AQI 50–200)
            with injected anomaly spikes (AQI 310–500) representing 13% of the dataset.
          </p>
        </div>
      </Card>
    </div>
  );
}
