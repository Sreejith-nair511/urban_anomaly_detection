"use client";
import { useState } from "react";
import { postPredict, PredictInput, PredictResult } from "../lib/api";
import { Card } from "./Card";
import { ConfidenceRadar } from "./Charts";
import { Zap, AlertTriangle, CheckCircle, AlertCircle } from "lucide-react";
import clsx from "clsx";

const ZONES = [
  "Koramangala","Whitefield","Indiranagar","Hebbal",
  "Electronic_City","Jayanagar","Marathahalli",
  "Yelahanka","BTM_Layout","HSR_Layout",
];

const DAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];

interface SliderProps {
  label: string; unit: string; min: number; max: number;
  value: number; onChange: (v: number) => void; color?: string;
}

function Slider({ label, unit, min, max, value, onChange, color = "cyan" }: SliderProps) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className={clsx("font-semibold", color === "red" ? "text-red-400" : color === "amber" ? "text-amber-400" : "text-cyan-400")}>
          {value} {unit}
        </span>
      </div>
      <input
        type="range" min={min} max={max} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
        style={{
          background: `linear-gradient(to right, ${color === "red" ? "#f87171" : color === "amber" ? "#fbbf24" : "#06b6d4"} ${pct}%, #1e293b ${pct}%)`,
        }}
      />
      <div className="flex justify-between text-[10px] text-slate-700">
        <span>{min}</span><span>{max}</span>
      </div>
    </div>
  );
}

const BADGE = {
  "Normal":       { cls: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30", icon: CheckCircle },
  "Moderate":     { cls: "bg-amber-500/10   text-amber-400   border-amber-500/30",   icon: AlertTriangle },
  "High Anomaly": { cls: "bg-red-500/10     text-red-400     border-red-500/30",     icon: AlertCircle },
};

export default function PredictionForm() {
  const [form, setForm] = useState<PredictInput>({
    AQI: 120, temperature: 28, humidity: 60,
    traffic_density: 45, noise_level: 65,
    hour: 9, day_of_week: 0,
  });
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState<string | null>(null);

  const set = (key: keyof PredictInput) => (v: number) =>
    setForm((f) => ({ ...f, [key]: v }));

  const handlePredict = async () => {
    setLoading(true); setError(null);
    try {
      const { data } = await postPredict(form);
      setResult(data);
    } catch {
      setError("Could not reach the API. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const radarData = result
    ? ["Normal", "Moderate", "High Anomaly"].map((l) => ({
        label: l,
        rf:  result.rf_confidence[l]  ?? 0,
        xgb: result.xgb_confidence[l] ?? 0,
      }))
    : [];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Input panel */}
      <Card>
        <h3 className="text-sm font-semibold text-slate-300 mb-5">Sensor Input</h3>
        <div className="space-y-5">
          <Slider label="AQI"             unit=""   min={50}  max={500} value={form.AQI}             onChange={set("AQI")}             color={form.AQI > 300 ? "red" : form.AQI > 150 ? "amber" : "cyan"} />
          <Slider label="Temperature"     unit="°C" min={15}  max={40}  value={form.temperature}     onChange={set("temperature")}     color="cyan" />
          <Slider label="Humidity"        unit="%"  min={30}  max={90}  value={form.humidity}        onChange={set("humidity")}        color="cyan" />
          <Slider label="Traffic Density" unit=""   min={0}   max={100} value={form.traffic_density} onChange={set("traffic_density")} color={form.traffic_density > 80 ? "red" : "amber"} />
          <Slider label="Noise Level"     unit="dB" min={30}  max={120} value={form.noise_level}     onChange={set("noise_level")}     color={form.noise_level > 80 ? "red" : "amber"} />

          <div className="grid grid-cols-2 gap-3 pt-1">
            <div>
              <label className="text-xs text-slate-400 block mb-1">Hour of Day</label>
              <select
                value={form.hour}
                onChange={(e) => setForm((f) => ({ ...f, hour: Number(e.target.value) }))}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-cyan-500"
              >
                {Array.from({ length: 24 }, (_, i) => (
                  <option key={i} value={i}>{String(i).padStart(2, "0")}:00</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs text-slate-400 block mb-1">Day of Week</label>
              <select
                value={form.day_of_week}
                onChange={(e) => setForm((f) => ({ ...f, day_of_week: Number(e.target.value) }))}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-cyan-500"
              >
                {DAYS.map((d, i) => <option key={i} value={i}>{d}</option>)}
              </select>
            </div>
          </div>

          <button
            onClick={handlePredict}
            disabled={loading}
            className={clsx(
              "w-full flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold transition-all duration-200",
              loading
                ? "bg-slate-700 text-slate-500 cursor-not-allowed"
                : "bg-gradient-to-r from-cyan-500 to-indigo-600 text-white hover:opacity-90 hover:shadow-[0_0_20px_rgba(6,182,212,0.3)]"
            )}
          >
            {loading ? (
              <><span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Predicting…</>
            ) : (
              <><Zap size={15} /> Predict Anomaly</>
            )}
          </button>

          {error && (
            <p className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
              {error}
            </p>
          )}
        </div>
      </Card>

      {/* Result panel */}
      <Card>
        <h3 className="text-sm font-semibold text-slate-300 mb-5">Prediction Result</h3>
        {result ? (
          <div className="space-y-5 animate-fade-in">
            {/* RF badge */}
            {(["rf", "xgb"] as const).map((m) => {
              const pred = m === "rf" ? result.rf_prediction : result.xgb_prediction;
              const b    = BADGE[pred as keyof typeof BADGE] ?? BADGE["Normal"];
              const Icon = b.icon;
              return (
                <div key={m} className="flex items-center justify-between">
                  <span className="text-xs text-slate-500 font-medium">
                    {m === "rf" ? "🌲 Random Forest" : "⚡ XGBoost"}
                  </span>
                  <span className={clsx("flex items-center gap-1.5 px-4 py-1.5 rounded-full text-sm font-bold border", b.cls)}>
                    <Icon size={14} />
                    {pred}
                  </span>
                </div>
              );
            })}

            <div className="border-t border-slate-800 pt-4">
              <p className="text-xs text-slate-500 mb-3">Model Confidence</p>
              <ConfidenceRadar data={radarData} />
            </div>

            {/* Confidence bars */}
            <div className="space-y-2">
              {["Normal", "Moderate", "High Anomaly"].map((l) => {
                const pct = (result.rf_confidence[l] ?? 0) * 100;
                const color = l === "Normal" ? "bg-emerald-500" : l === "Moderate" ? "bg-amber-500" : "bg-red-500";
                return (
                  <div key={l}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-400">{l}</span>
                      <span className="text-slate-300">{pct.toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div className={clsx("h-full rounded-full transition-all duration-700", color)} style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-64 text-slate-600">
            <Zap size={40} className="mb-3 opacity-30" />
            <p className="text-sm">Adjust sliders and click</p>
            <p className="text-sm font-semibold text-cyan-600 mt-1">Predict Anomaly</p>
          </div>
        )}
      </Card>
    </div>
  );
}
