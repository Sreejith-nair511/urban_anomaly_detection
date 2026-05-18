"use client";
import { useEffect, useState } from "react";
import { Brain, Zap, AlertTriangle, TrendingUp, Clock } from "lucide-react";
import clsx from "clsx";

const INSIGHTS = [
  { severity: "critical", zone: "Marathahalli",    icon: "🚨", msg: "Heavy congestion predicted in next 30 min. Vehicle count exceeds baseline by 340%. AI confidence: 94%." },
  { severity: "critical", zone: "Whitefield",      icon: "☁",  msg: "AQI spike detected near industrial sector. PM2.5 at 68 µg/m³. Pattern matches diesel generator activity." },
  { severity: "warning",  zone: "HSR Layout",      icon: "🔊", msg: "Noise anomaly — 104dB sustained. Pattern correlates with construction machinery. Duration: 4+ hours." },
  { severity: "warning",  zone: "Silk Board",      icon: "🚦", msg: "Traffic density at 91%. Multi-lane blockage pattern detected. Recommend signal phase optimization." },
  { severity: "warning",  zone: "Electronic City", icon: "🌡", msg: "Thermal anomaly in Phase 2. Ambient temp 39°C. Cross-referenced with AQI spike — likely industrial event." },
  { severity: "normal",   zone: "Jayanagar",       icon: "✅", msg: "All parameters within normal bounds. AQI 72, Traffic 28%, Noise 44dB. Zone stable." },
  { severity: "critical", zone: "Hebbal Flyover",  icon: "⚡", msg: "Combined anomaly: AQI 312, Traffic 89%, Noise 97dB simultaneously. High Anomaly class — immediate review." },
  { severity: "normal",   zone: "Yelahanka",       icon: "✅", msg: "Night baseline confirmed normal. No anomalous signatures in last 45 minutes." },
  { severity: "warning",  zone: "BTM Layout",      icon: "💨", msg: "Humidity spike 87% combined with AQI 195. Pattern suggests waterlogged road debris — noise risk." },
  { severity: "critical", zone: "Koramangala",     icon: "🔴", msg: "Multi-parameter breach. All 5 sensors elevated simultaneously. Possible large-scale outdoor event." },
];

const SEV = {
  critical: { color: "text-red-400",   border: "border-red-500/20",   bg: "bg-red-500/5",   dot: "bg-red-400"    },
  warning:  { color: "text-amber-400", border: "border-amber-500/20", bg: "bg-amber-500/5", dot: "bg-amber-400"  },
  normal:   { color: "text-green-400", border: "border-green-500/20", bg: "bg-green-500/5", dot: "bg-green-400"  },
};

export default function AIInsights() {
  const [current, setCurrent] = useState(0);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      setVisible(false);
      setTimeout(() => {
        setCurrent(c => (c + 1) % INSIGHTS.length);
        setVisible(true);
      }, 300);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="glass-card rounded-2xl p-4 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4 pb-3 border-b border-slate-800/60">
        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center">
          <Brain size={13} className="text-white" />
        </div>
        <div>
          <p className="text-[10px] font-bold text-slate-300 font-mono tracking-widest">AI URBAN INSIGHTS</p>
          <p className="text-[9px] text-purple-500 font-mono">Real-time pattern intelligence</p>
        </div>
        <div className="ml-auto flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-pulse" />
          <span className="text-[9px] text-purple-500 font-mono">LIVE</span>
        </div>
      </div>

      {/* Rotating insight */}
      <div
        className={clsx(
          "flex-1 transition-all duration-300",
          visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2"
        )}
      >
        {(() => {
          const ins = INSIGHTS[current];
          const s = SEV[ins.severity as keyof typeof SEV];
          return (
            <div className={clsx("rounded-xl border p-3 mb-3", s.border, s.bg)}>
              <div className="flex items-start gap-2">
                <span className="text-lg">{ins.icon}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={clsx("text-[9px] font-bold font-mono tracking-wider", s.color)}>
                      {ins.severity.toUpperCase()}
                    </span>
                    <span className="text-[9px] text-slate-600 font-mono">·</span>
                    <span className="text-[9px] text-slate-400 font-mono">{ins.zone}</span>
                  </div>
                  <p className="text-[11px] text-slate-300 leading-relaxed">{ins.msg}</p>
                </div>
              </div>
            </div>
          );
        })()}
      </div>

      {/* All insights list */}
      <div className="space-y-1.5 overflow-y-auto max-h-48">
        {INSIGHTS.slice(0, 6).map((ins, i) => {
          const s = SEV[ins.severity as keyof typeof SEV];
          return (
            <div key={i}
              className={clsx(
                "flex items-start gap-2 px-2 py-1.5 rounded-lg cursor-pointer transition-all duration-200",
                i === current % 6 ? clsx("border", s.border, s.bg) : "hover:bg-slate-800/30"
              )}
              onClick={() => { setCurrent(i); setVisible(true); }}
            >
              <span className={clsx("w-1.5 h-1.5 rounded-full flex-shrink-0 mt-1", s.dot)} />
              <div className="flex-1 min-w-0">
                <span className="text-[9px] text-slate-500 font-mono">{ins.zone}</span>
                <p className="text-[10px] text-slate-400 truncate">{ins.msg.substring(0, 55)}…</p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="mt-3 pt-3 border-t border-slate-800/60 flex items-center gap-2">
        <Clock size={9} className="text-slate-700" />
        <span className="text-[9px] text-slate-700 font-mono">Updated every 4s · DBSCAN + RF inference</span>
      </div>
    </div>
  );
}
