"use client";
import { useState, useEffect } from "react";
import { postPredict, PredictInput, PredictResult } from "../../lib/api";
import { Card, SectionTitle } from "../../components/Card";
import { ConfidenceRadar } from "../../components/Charts";
import { Zap, AlertTriangle, CheckCircle, AlertCircle, Brain, Target, Activity, Clock } from "lucide-react";
import clsx from "clsx";

const ZONES = ["Koramangala","Whitefield","Indiranagar","Hebbal","Electronic_City","Jayanagar","Marathahalli","Yelahanka","BTM_Layout","HSR_Layout"];
const DAYS  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];

const SCENARIOS = [
  { name: "Rush Hour Gridlock",   form: { AQI:220, temperature:36, humidity:72, traffic_density:92, noise_level:95, hour:8,  day_of_week:1 } },
  { name: "Festival Traffic",     form: { AQI:180, temperature:34, humidity:65, traffic_density:98, noise_level:110,hour:20, day_of_week:6 } },
  { name: "Industrial Pollution", form: { AQI:420, temperature:38, humidity:80, traffic_density:55, noise_level:78, hour:14, day_of_week:3 } },
  { name: "Night Normal",         form: { AQI: 80, temperature:26, humidity:55, traffic_density:15, noise_level:42, hour:2,  day_of_week:2 } },
  { name: "Construction Noise",   form: { AQI:155, temperature:33, humidity:62, traffic_density:60, noise_level:108,hour:10, day_of_week:0 } },
  { name: "Rain Congestion",      form: { AQI:200, temperature:28, humidity:90, traffic_density:85, noise_level:88, hour:18, day_of_week:4 } },
];

const BADGE = {
  "Normal":       { cls:"bg-green-500/10  text-green-400  border-green-500/30",  icon: CheckCircle, glow:"shadow-[0_0_20px_rgba(16,185,129,0.3)]"  },
  "Moderate":     { cls:"bg-amber-500/10  text-amber-400  border-amber-500/30",  icon: AlertTriangle,glow:"shadow-[0_0_20px_rgba(245,158,11,0.3)]"  },
  "High Anomaly": { cls:"bg-red-500/10    text-red-400    border-red-500/30",    icon: AlertCircle,  glow:"shadow-[0_0_20px_rgba(239,68,68,0.3)]"   },
};

const RECOMMEND: Record<string, string[]> = {
  "Normal":       ["✅ No action required.", "📊 Continue standard monitoring.", "🟢 All parameters within safe bounds."],
  "Moderate":     ["⚠ Alert zone operators.", "📡 Increase sensor polling frequency.", "🚦 Prepare signal optimization response.", "📱 Send advisory to traffic control."],
  "High Anomaly": ["🚨 IMMEDIATE action required.", "📞 Dispatch emergency response teams.", "🔴 Trigger zone-wide alert notifications.", "🚦 Activate emergency signal protocols.", "📰 Issue public advisory for this zone."],
};

interface SliderProps { label: string; unit: string; min: number; max: number; value: number; onChange:(v:number)=>void; color?:string; icon?: React.ReactNode; }
function Slider({ label, unit, min, max, value, onChange, color="cyan", icon }: SliderProps) {
  const pct = ((value - min) / (max - min)) * 100;
  const colorHex = color === "red" ? "#ef4444" : color === "amber" ? "#f59e0b" : "#06b6d4";
  const risk = pct > 80 ? "CRITICAL" : pct > 60 ? "HIGH" : pct > 40 ? "MODERATE" : "LOW";
  const riskColor = pct > 80 ? "text-red-400" : pct > 60 ? "text-amber-400" : pct > 40 ? "text-amber-300" : "text-green-400";

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {icon && <span style={{ color: colorHex }}>{icon}</span>}
          <span className="text-[11px] text-slate-400 font-mono">{label}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-[8px] font-bold font-mono ${riskColor}`}>{risk}</span>
          <span className="text-sm font-black font-mono" style={{ color: colorHex }}>{value}<span className="text-xs text-slate-500 font-normal ml-0.5">{unit}</span></span>
        </div>
      </div>
      <div className="relative">
        <input type="range" min={min} max={max} value={value}
          onChange={e => onChange(Number(e.target.value))}
          className="w-full h-2 rounded-full appearance-none cursor-pointer"
          style={{ background: `linear-gradient(to right, ${colorHex} ${pct}%, #1e293b ${pct}%)` }}
        />
      </div>
      <div className="flex justify-between text-[9px] text-slate-700 font-mono">
        <span>{min}{unit}</span><span>{max}{unit}</span>
      </div>
    </div>
  );
}

// Circular gauge component
function Gauge({ value, label, color }: { value: number; label: string; color: string }) {
  const circumference = 2 * Math.PI * 36;
  const dash = (value / 100) * circumference;
  return (
    <div className="flex flex-col items-center">
      <div className="relative w-20 h-20">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 80 80">
          <circle cx="40" cy="40" r="36" fill="none" stroke="rgba(30,41,59,0.8)" strokeWidth="6" />
          <circle cx="40" cy="40" r="36" fill="none" stroke={color} strokeWidth="6"
            strokeDasharray={`${dash} ${circumference}`} strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 6px ${color})`, transition: "stroke-dasharray 0.6s ease" }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-black font-mono" style={{ color }}>{Math.round(value)}%</span>
        </div>
      </div>
      <p className="text-[9px] text-slate-500 font-mono mt-1 text-center">{label}</p>
    </div>
  );
}

export default function IncidentAnalyzer() {
  const [form, setForm] = useState<PredictInput>({
    AQI: 120, temperature: 28, humidity: 60, traffic_density: 45, noise_level: 65, hour: 9, day_of_week: 0,
  });
  const [result, setResult]   = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);
  const [inferenceMs, setInferenceMs] = useState<number | null>(null);

  const set = (key: keyof PredictInput) => (v: number) => setForm(f => ({ ...f, [key]: v }));

  const loadScenario = (s: typeof SCENARIOS[0]) => {
    setForm(f => ({ ...f, ...s.form }));
    setResult(null);
  };

  const handlePredict = async () => {
    setLoading(true); setError(null);
    const t0 = performance.now();
    try {
      const { data } = await postPredict(form);
      setResult(data);
      setInferenceMs(Math.round(performance.now() - t0));
    } catch {
      setError("Cannot reach FastAPI backend at port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const radarData = result
    ? ["Normal","Moderate","High Anomaly"].map(l => ({ label:l, rf: result.rf_confidence[l]??0, xgb: result.xgb_confidence[l]??0 }))
    : [];

  // Composite risk score (weighted)
  const riskScore = Math.min(100, Math.round(
    (form.AQI / 500) * 35 + (form.traffic_density / 100) * 35 + (form.noise_level / 120) * 20 + ((form.temperature - 15) / 25) * 10
  ));
  const riskColor = riskScore > 70 ? "#ef4444" : riskScore > 45 ? "#f59e0b" : "#10b981";
  const riskLabel = riskScore > 70 ? "HIGH RISK" : riskScore > 45 ? "MODERATE RISK" : "LOW RISK";

  const aqiRisk      = Math.round((form.AQI / 500) * 100);
  const trafficRisk  = form.traffic_density;
  const noiseRisk    = Math.round((form.noise_level / 120) * 100);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-red-500 to-rose-700 flex items-center justify-center shadow-neon-red">
          <Brain size={16} className="text-white" />
        </div>
        <div>
          <h1 className="text-xl font-black text-white" style={{ fontFamily: "'Orbitron', sans-serif" }}>AI INCIDENT ANALYZER</h1>
          <p className="text-[10px] text-slate-500 font-mono">Real-time urban anomaly prediction · Random Forest + XGBoost</p>
        </div>
      </div>

      {/* Preset Scenarios */}
      <div>
        <p className="text-[9px] text-slate-600 font-mono uppercase tracking-widest mb-2">▸ QUICK SCENARIOS</p>
        <div className="flex flex-wrap gap-2">
          {SCENARIOS.map(s => (
            <button key={s.name} onClick={() => loadScenario(s)}
              className="text-[9px] font-bold font-mono px-3 py-1.5 rounded-lg border border-slate-700/60 bg-slate-800/30 text-slate-400 hover:text-cyan-400 hover:border-cyan-500/40 hover:bg-cyan-500/5 transition-all">
              {s.name}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Input Panel ── */}
        <Card glow="cyan">
          <SectionTitle>SENSOR INPUT PARAMETERS</SectionTitle>
          <div className="space-y-5">
            <Slider label="Air Quality Index (AQI)" unit="" min={50} max={500} value={form.AQI} onChange={set("AQI")}
              color={form.AQI > 300 ? "red" : form.AQI > 150 ? "amber" : "cyan"}
              icon={<Activity size={11}/>} />
            <Slider label="Ambient Temperature" unit="°C" min={15} max={40} value={form.temperature} onChange={set("temperature")}
              color={form.temperature > 37 ? "red" : "cyan"} />
            <Slider label="Relative Humidity" unit="%" min={30} max={90} value={form.humidity} onChange={set("humidity")}
              color={form.humidity > 80 ? "amber" : "cyan"} />
            <Slider label="Traffic Density" unit="%" min={0} max={100} value={form.traffic_density} onChange={set("traffic_density")}
              color={form.traffic_density > 80 ? "red" : form.traffic_density > 60 ? "amber" : "cyan"}
              icon={<Target size={11}/>} />
            <Slider label="Noise Level" unit=" dB" min={30} max={120} value={form.noise_level} onChange={set("noise_level")}
              color={form.noise_level > 90 ? "red" : form.noise_level > 70 ? "amber" : "cyan"} />

            <div className="grid grid-cols-2 gap-3 pt-1">
              {[
                { label:"Hour of Day", key:"hour" as keyof PredictInput, opts: Array.from({length:24},(_,i)=>({v:i,label:`${String(i).padStart(2,"0")}:00`})) },
                { label:"Day of Week",  key:"day_of_week" as keyof PredictInput, opts: DAYS.map((d,i)=>({v:i,label:d})) },
              ].map(sel => (
                <div key={sel.key}>
                  <label className="text-[9px] text-slate-500 font-mono uppercase tracking-wider block mb-1">{sel.label}</label>
                  <select value={form[sel.key]}
                    onChange={e => setForm(f => ({ ...f, [sel.key]: Number(e.target.value) }))}
                    className="w-full bg-slate-900 border border-slate-700/60 rounded-xl px-3 py-2 text-xs text-slate-200 font-mono focus:outline-none focus:border-cyan-500/60 focus:shadow-[0_0_12px_rgba(6,182,212,0.2)] transition-all">
                    {sel.opts.map(o => <option key={o.v} value={o.v}>{o.label}</option>)}
                  </select>
                </div>
              ))}
            </div>

            {/* Risk gauges */}
            <div className="pt-2 border-t border-slate-800/60">
              <p className="text-[9px] text-slate-600 font-mono mb-3">▸ REAL-TIME RISK METERS</p>
              <div className="flex items-center justify-around">
                <Gauge value={aqiRisk}     label="AIR QUALITY" color={aqiRisk>70?"#ef4444":aqiRisk>40?"#f59e0b":"#10b981"} />
                <Gauge value={trafficRisk} label="TRAFFIC"     color={trafficRisk>70?"#ef4444":trafficRisk>40?"#f59e0b":"#10b981"} />
                <Gauge value={noiseRisk}   label="NOISE"       color={noiseRisk>70?"#ef4444":noiseRisk>40?"#f59e0b":"#10b981"} />
                <Gauge value={riskScore}   label="COMPOSITE"   color={riskColor} />
              </div>
            </div>

            <button onClick={handlePredict} disabled={loading}
              className={clsx("w-full flex items-center justify-center gap-2 py-3.5 rounded-xl text-sm font-black transition-all duration-200 tracking-wider",
                loading
                  ? "bg-slate-800 text-slate-600 cursor-not-allowed"
                  : "bg-gradient-to-r from-cyan-500 via-blue-600 to-indigo-600 text-white hover:opacity-90 shadow-neon-cyan hover:shadow-[0_0_30px_rgba(6,182,212,0.5)]"
              )} style={{ fontFamily:"'Orbitron',sans-serif", fontSize:"11px" }}>
              {loading
                ? <><span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> ANALYZING...</>
                : <><Zap size={14}/> RUN AI ANALYSIS</>}
            </button>

            {error && <p className="text-[10px] text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2 font-mono">{error}</p>}
          </div>
        </Card>

        {/* ── Result Panel ── */}
        <div className="space-y-4">
          {result ? (
            <>
              {/* Composite risk */}
              <div className="glass-card rounded-2xl p-4 border" style={{ borderColor: riskColor+"40", background: riskColor+"08" }}>
                <div className="flex items-center justify-between mb-3">
                  <p className="text-[9px] font-bold font-mono text-slate-500 tracking-widest">INCIDENT RISK ASSESSMENT</p>
                  {inferenceMs && <span className="text-[9px] font-mono text-slate-600"><Clock size={8} className="inline mr-1"/>{inferenceMs}ms</span>}
                </div>
                <div className="flex items-center gap-4">
                  <Gauge value={riskScore} label={riskLabel} color={riskColor} />
                  <div className="flex-1">
                    {["rf","xgb"].map(m => {
                      const pred = m === "rf" ? result.rf_prediction : result.xgb_prediction;
                      const b = BADGE[pred as keyof typeof BADGE] ?? BADGE["Normal"];
                      const Icon = b.icon;
                      return (
                        <div key={m} className="flex items-center justify-between mb-2">
                          <span className="text-[10px] text-slate-500 font-mono">{m === "rf" ? "🌲 Random Forest" : "⚡ XGBoost"}</span>
                          <span className={clsx("flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-bold border", b.cls, b.glow)}>
                            <Icon size={11}/>{pred}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* Confidence bars */}
              <Card>
                <SectionTitle>MODEL CONFIDENCE BREAKDOWN</SectionTitle>
                <div className="space-y-3">
                  {["Normal","Moderate","High Anomaly"].map(l => {
                    const rf  = (result.rf_confidence[l]  ?? 0) * 100;
                    const xgb = (result.xgb_confidence[l] ?? 0) * 100;
                    const barColor = l==="Normal" ? "bg-green-500" : l==="Moderate" ? "bg-amber-500" : "bg-red-500";
                    const barColorXgb = l==="Normal" ? "bg-green-400" : l==="Moderate" ? "bg-amber-400" : "bg-red-400";
                    return (
                      <div key={l}>
                        <div className="flex justify-between text-[9px] mb-1 font-mono">
                          <span className="text-slate-400">{l}</span>
                          <span className="text-slate-500">RF: {rf.toFixed(1)}% · XGB: {xgb.toFixed(1)}%</span>
                        </div>
                        <div className="space-y-1">
                          <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                            <div className={clsx("h-full rounded-full transition-all duration-700", barColor)} style={{ width:`${rf}%` }} />
                          </div>
                          <div className="h-1 bg-slate-800/60 rounded-full overflow-hidden">
                            <div className={clsx("h-full rounded-full transition-all duration-700 opacity-60", barColorXgb)} style={{ width:`${xgb}%` }} />
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="mt-3 pt-3 border-t border-slate-800/60">
                  <ConfidenceRadar data={radarData} />
                </div>
              </Card>

              {/* AI Recommendations */}
              <Card glow={result.rf_prediction === "High Anomaly" ? "red" : result.rf_prediction === "Moderate" ? "amber" : "emerald"}>
                <SectionTitle>AI RESPONSE RECOMMENDATIONS</SectionTitle>
                <div className="space-y-2">
                  {(RECOMMEND[result.rf_prediction] ?? RECOMMEND["Normal"]).map((rec, i) => (
                    <div key={i} className="flex items-start gap-2 p-2.5 rounded-lg bg-slate-800/30 border border-slate-700/30">
                      <p className="text-[11px] text-slate-300">{rec}</p>
                    </div>
                  ))}
                </div>
              </Card>
            </>
          ) : (
            <Card className="flex flex-col items-center justify-center h-80">
              <div className="relative w-16 h-16 mb-4">
                <div className="absolute inset-0 rounded-full border-2 border-cyan-500/20 animate-spin-slow" />
                <div className="absolute inset-2 rounded-full border border-cyan-500/10" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <Brain size={24} className="text-cyan-600" />
                </div>
              </div>
              <p className="text-sm font-bold text-slate-500 font-mono">AWAITING ANALYSIS</p>
              <p className="text-[10px] text-slate-700 font-mono mt-1">Configure parameters and run AI analysis</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
