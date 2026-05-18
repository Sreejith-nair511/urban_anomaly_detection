"use client";
import { useEffect, useState } from "react";
import { fetchMetrics, fetchSummary, Metrics, DataSummary } from "../lib/api";
import { KpiCard, Card, SectionTitle, Spinner, ErrorMsg } from "../components/Card";
import { ZoneAnomalyBar, HourlyTrend, AqiHistogram } from "../components/Charts";
import AIInsights from "../components/AIInsights";
import BangaloreMap from "../components/BangaloreMap";
import CCTVPanel from "../components/CCTVPanel";
import {
  AlertTriangle, Activity, Camera, Radio,
  Cpu, Database, Shield, Wind, Zap, Eye,
} from "lucide-react";

// Simulated live anomaly event log
const EVENT_LOG = [
  { time: "19:54:02", zone: "Marathahalli",    event: "High Anomaly detected — RF+XGB consensus",   sev: "critical" },
  { time: "19:52:41", zone: "Whitefield",       event: "AQI crossed 350 threshold — Type 1 spike",   sev: "critical" },
  { time: "19:51:18", zone: "Hebbal",           event: "Traffic density 91% — surge event active",   sev: "critical" },
  { time: "19:49:55", zone: "HSR Layout",       event: "Noise anomaly 104dB — construction pattern", sev: "warning"  },
  { time: "19:48:33", zone: "BTM Layout",       event: "Humidity-AQI correlation spike detected",    sev: "warning"  },
  { time: "19:47:10", zone: "Indiranagar",      event: "Moderate anomaly — traffic 55%, noise 72dB", sev: "warning"  },
  { time: "19:45:02", zone: "Jayanagar",        event: "Zone cleared — all parameters normal",       sev: "normal"   },
  { time: "19:44:30", zone: "Yelahanka",        event: "Night baseline confirmed stable",             sev: "normal"   },
];

const CCTV_FEEDS = [
  {
    videoSrc: "/videos/v1.mp4",
    cameraId: "CAM-01 · SILK BOARD",
    location: "Silk Board Junction, Outer Ring Road",
    status: "critical" as const,
    detections: [
      { label: "HIGH TRAFFIC",    top: "20%", left: "10%", width: "35%", height: "50%", color: "#ef4444" },
      { label: "CONGESTION ZONE", top: "40%", left: "55%", width: "30%", height: "35%", color: "#f59e0b" },
    ],
  },
  {
    videoSrc: "/videos/v2.mp4",
    cameraId: "CAM-04 · WHITEFIELD",
    location: "Whitefield Main Rd, Industrial Sector",
    status: "warning" as const,
    detections: [
      { label: "POLLUTION SPIKE", top: "15%", left: "25%", width: "40%", height: "30%", color: "#f59e0b" },
      { label: "MODERATE TRAFFIC",top: "55%", left: "10%", width: "45%", height: "35%", color: "#f59e0b" },
    ],
  },
  {
    videoSrc: "/videos/v3.mp4",
    cameraId: "CAM-07 · MARATHAHALLI",
    location: "Marathahalli Bridge, ORR",
    status: "critical" as const,
    detections: [
      { label: "COMBINED ANOMALY",top: "20%", left: "15%", width: "70%", height: "60%", color: "#ef4444" },
      { label: "NOISE EVENT",     top: "70%", left: "60%", width: "30%", height: "20%", color: "#ef4444" },
    ],
  },
];

export default function CommandCenter() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [loading, setLoading]  = useState(true);
  const [error, setError]      = useState<string | null>(null);
  const [selectedZone, setSelectedZone] = useState<any>(null);

  useEffect(() => {
    Promise.all([fetchMetrics(), fetchSummary()])
      .then(([m, s]) => { setMetrics(m.data); setSummary(s.data); })
      .catch(() => setError("Backend offline. Start the FastAPI server on port 8000."))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Spinner />;
  if (error)   return <ErrorMsg msg={error} />;
  if (!metrics || !summary) return null;

  const dist = summary.label_distribution;
  const highAnomaly  = dist["High Anomaly"] ?? 0;
  const moderate     = dist["Moderate"] ?? 0;
  const totalAnomaly = highAnomaly + moderate;
  const rfAcc        = (metrics.random_forest.accuracy * 100).toFixed(1);
  const xgbAcc       = (metrics.xgboost.accuracy * 100).toFixed(1);

  return (
    <div className="space-y-6 animate-fade-in">

      {/* ── Page Header ── */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-700 flex items-center justify-center shadow-neon-cyan">
              <Shield size={15} className="text-white" />
            </div>
            <h1 className="text-xl font-black text-white" style={{ fontFamily: "'Orbitron', sans-serif" }}>
              COMMAND CENTER
            </h1>
            <span className="text-[9px] font-bold font-mono bg-green-500/10 text-green-400 border border-green-500/30 px-2 py-1 rounded animate-pulse">
              ● LIVE MONITORING
            </span>
          </div>
          <p className="text-[11px] text-slate-500 font-mono ml-11">
            Bangalore Urban Anomaly Intelligence · 10 Zones · {summary.total_samples.toLocaleString()} Sensor Records Analyzed
          </p>
        </div>
        <div className="text-right">
          <p className="text-[9px] text-slate-600 font-mono">ML MODELS</p>
          <p className="text-xs font-bold text-cyan-400 font-mono">RF {rfAcc}% · XGB {xgbAcc}%</p>
          <p className="text-[9px] text-slate-700 font-mono">DBSCAN · PCA · K-Means</p>
        </div>
      </div>

      {/* ── KPI Row 1: Primary Alerts ── */}
      <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 gap-3">
        <KpiCard icon={<AlertTriangle size={16}/>}  label="Active Anomalies"   value={totalAnomaly}              unit="" sub={`${((totalAnomaly/summary.total_samples)*100).toFixed(1)}% of dataset`} color="red"     trend="up"   />
        <KpiCard icon={<Wind size={16}/>}           label="Pollution Alerts"   value={highAnomaly}               unit="" sub="AQI > 300 events"                                                        color="amber"   trend="up"   />
        <KpiCard icon={<Activity size={16}/>}       label="Congested Zones"    value={4}                         unit="" sub="ORR · Silk Board · Hebbal"                                               color="red"     trend="up"   />
        <KpiCard icon={<Eye size={16}/>}            label="Live Cameras"        value={7}                         unit="" sub="3 critical · 4 normal"                                                  color="purple"  trend="flat" />
        <KpiCard icon={<Database size={16}/>}       label="Monitored Sensors"  value={summary.total_samples}     unit="" sub="5,600 records active"                                                    color="cyan"    trend="flat" />
        <KpiCard icon={<Cpu size={16}/>}            label="AI Risk Level"       value={74}                        unit="%" sub="Composite risk score"                                                   color="red"     trend="up"   />
      </div>

      {/* ── KPI Row 2: Model Metrics ── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: "RF Accuracy",  val: `${rfAcc}%`,  sub: `F1: ${metrics.random_forest.f1_score}`,  color: "emerald" as const },
          { label: "XGB Accuracy", val: `${xgbAcc}%`, sub: `F1: ${metrics.xgboost.f1_score}`,        color: "indigo"  as const },
          { label: "Silhouette",   val: metrics.kmeans_silhouette.toFixed(3), sub: "K-Means cluster quality", color: "cyan" as const },
          { label: "PCA Variance", val: `${((metrics.pca_variance[0]+metrics.pca_variance[1])*100).toFixed(1)}%`, sub: "PC1+PC2 explained", color: "purple" as const },
        ].map(m => (
          <div key={m.label} className="glass-card rounded-xl p-4 flex items-center justify-between">
            <div>
              <p className="text-[9px] text-slate-600 font-mono uppercase tracking-wider">{m.label}</p>
              <p className={`text-lg font-black font-mono text-${m.color}-400 mt-0.5`}>{m.val}</p>
            </div>
            <p className="text-[9px] text-slate-600 font-mono text-right">{m.sub}</p>
          </div>
        ))}
      </div>

      {/* ── Map + AI Insights ── */}
      <div className="grid grid-cols-1 xl:grid-cols-5 gap-4" style={{ minHeight: "440px" }}>
        <div className="xl:col-span-3">
          <BangaloreMap onZoneClick={setSelectedZone} />
        </div>
        <div className="xl:col-span-2">
          <AIInsights />
        </div>
      </div>

      {/* ── CCTV Surveillance Grid ── */}
      <div>
        <SectionTitle>LIVE SURVEILLANCE FEEDS</SectionTitle>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {CCTV_FEEDS.map((feed, i) => (
            <CCTVPanel key={i} {...feed} />
          ))}
        </div>
      </div>

      {/* ── Zone Chart + Event Log ── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Card glow="cyan">
          <SectionTitle>ZONE ANOMALY RATES</SectionTitle>
          <ZoneAnomalyBar data={summary.zone_anomaly_rates} />
        </Card>

        {/* AI Event Log */}
        <Card glow="red">
          <SectionTitle>AI EVENT LOG</SectionTitle>
          <div className="space-y-1.5 overflow-y-auto max-h-[240px]">
            {EVENT_LOG.map((ev, i) => (
              <div key={i} className={`flex items-start gap-3 px-3 py-2 rounded-lg ${
                ev.sev === "critical" ? "bg-red-500/5 border border-red-500/15" :
                ev.sev === "warning"  ? "bg-amber-500/5 border border-amber-500/15" :
                                        "bg-green-500/5 border border-green-500/15"
              }`}>
                <span className={`mt-0.5 w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                  ev.sev === "critical" ? "bg-red-400 animate-pulse" :
                  ev.sev === "warning"  ? "bg-amber-400" : "bg-green-400"
                }`}/>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-[9px] text-slate-600 font-mono">{ev.time}</span>
                    <span className={`text-[9px] font-bold font-mono ${
                      ev.sev === "critical" ? "text-red-400" : ev.sev === "warning" ? "text-amber-400" : "text-green-400"
                    }`}>{ev.zone}</span>
                  </div>
                  <p className="text-[10px] text-slate-400 truncate">{ev.event}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* ── Hourly Trend + AQI Histogram ── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Card glow="cyan">
          <SectionTitle>24H SENSOR TRENDS</SectionTitle>
          <HourlyTrend data={summary.hourly_trends} />
        </Card>
        <Card glow="amber">
          <SectionTitle>AQI DISTRIBUTION</SectionTitle>
          <AqiHistogram data={summary.aqi_histogram} />
        </Card>
      </div>

      {/* ── Label Distribution ── */}
      <Card glow="cyan">
        <SectionTitle>ANOMALY CLASS DISTRIBUTION</SectionTitle>
        <div className="grid grid-cols-3 gap-4">
          {Object.entries(dist).map(([label, count]) => {
            const pct = ((count / summary.total_samples) * 100).toFixed(1);
            const cfg = label === "Normal"
              ? { color: "emerald", bar: "bg-emerald-500", text: "text-emerald-400", glow: "shadow-[0_0_12px_rgba(16,185,129,0.4)]" }
              : label === "Moderate"
              ? { color: "amber",   bar: "bg-amber-500",   text: "text-amber-400",   glow: "shadow-[0_0_12px_rgba(245,158,11,0.4)]" }
              : { color: "red",     bar: "bg-red-500",     text: "text-red-400",     glow: "shadow-[0_0_12px_rgba(239,68,68,0.4)]"  };
            return (
              <div key={label} className={`rounded-xl border p-4 bg-${cfg.color}-500/5 border-${cfg.color}-500/20 ${cfg.glow} transition-all hover:scale-[1.02]`}>
                <p className="text-[9px] text-slate-500 font-mono uppercase tracking-widest mb-1">{label}</p>
                <p className={`text-3xl font-black ${cfg.text}`} style={{ fontFamily: "'Orbitron', sans-serif" }}>
                  {count.toLocaleString()}
                </p>
                <div className="mt-3 h-1 bg-slate-800 rounded-full overflow-hidden">
                  <div className={`h-full ${cfg.bar} rounded-full transition-all duration-1000`} style={{ width: `${pct}%` }} />
                </div>
                <p className="text-[10px] text-slate-600 mt-1 font-mono">{pct}% of dataset</p>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}
