"use client";
import { useEffect, useRef } from "react";
import { AlertTriangle, AlertCircle, CheckCircle, Radio } from "lucide-react";

const ALERTS = [
  { type: "critical", text: "⚡ HIGH CONGESTION DETECTED — Silk Board Junction, ORR Corridor" },
  { type: "warning",  text: "☁ AQI SPIKE: Whitefield Industrial Sector — AQI 387 (Hazardous)" },
  { type: "warning",  text: "🔊 NOISE EVENT: HSR Layout — 104dB — Likely Construction Activity" },
  { type: "normal",   text: "✅ ZONE CLEAR: Jayanagar — All Parameters Normal" },
  { type: "critical", text: "🚨 TRAFFIC SURGE: Hebbal Flyover — Density 94% — Evening Rush" },
  { type: "warning",  text: "🌡 THERMAL ANOMALY: Electronic City Phase 2 — Temp 39°C" },
  { type: "normal",   text: "✅ ZONE CLEAR: Yelahanka — AQI 72, Traffic 34% — Stable" },
  { type: "critical", text: "⚠ COMBINED ANOMALY: Marathahalli — AQI 312, Traffic 89%, Noise 97dB" },
  { type: "warning",  text: "🔴 POLLUTION ALERT: BTM Layout — PM2.5 Elevated — 68 µg/m³" },
  { type: "normal",   text: "✅ ZONE CLEAR: Indiranagar — Night Baseline Normal" },
  { type: "critical", text: "🚨 EMERGENCY: Koramangala Sector 7 — Multi-Parameter Breach Detected" },
];

const COLOR = {
  critical: "text-red-400",
  warning:  "text-amber-400",
  normal:   "text-green-400",
};

export default function Ticker() {
  return (
    <div className="h-8 flex items-center border-b border-cyan-900/30 overflow-hidden"
      style={{ background: "rgba(2,6,23,0.95)" }}>
      {/* Label */}
      <div className="flex-shrink-0 flex items-center gap-2 px-4 border-r border-cyan-900/40 h-full bg-cyan-500/5">
        <Radio size={10} className="text-cyan-400 animate-pulse" />
        <span className="text-[9px] font-bold font-mono text-cyan-400 tracking-widest whitespace-nowrap">LIVE INTEL</span>
      </div>
      {/* Scrolling content */}
      <div className="ticker-wrap flex-1 overflow-hidden">
        <div className="ticker-content flex items-center gap-8 py-1">
          {[...ALERTS, ...ALERTS].map((a, i) => (
            <span key={i} className={`text-[10px] font-mono font-medium whitespace-nowrap ${COLOR[a.type as keyof typeof COLOR]}`}>
              {a.text}
              <span className="text-slate-700 ml-6">◆</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
