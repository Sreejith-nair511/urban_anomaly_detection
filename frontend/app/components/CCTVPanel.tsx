"use client";
import { useEffect, useRef, useState } from "react";
import clsx from "clsx";

interface CCTVPanelProps {
  videoSrc: string;
  cameraId: string;
  location: string;
  status: "normal" | "warning" | "critical";
  detections?: Array<{ label: string; top: string; left: string; width: string; height: string; color: string; }>;
}

const STATUS_CONFIG = {
  normal:   { label: "NORMAL ZONE",       color: "text-green-400", border: "border-green-500/40",  bg: "bg-green-500/10"  },
  warning:  { label: "MODERATE ACTIVITY", color: "text-amber-400", border: "border-amber-500/40",  bg: "bg-amber-500/10"  },
  critical: { label: "HIGH ANOMALY",      color: "text-red-400",   border: "border-red-500/40",    bg: "bg-red-500/10"    },
};

export default function CCTVPanel({ videoSrc, cameraId, location, status, detections = [] }: CCTVPanelProps) {
  const [time, setTime] = useState("");
  const [showBoxes, setShowBoxes] = useState(true);
  const cfg = STATUS_CONFIG[status];

  useEffect(() => {
    const i = setInterval(() => {
      setTime(new Date().toLocaleTimeString("en-IN", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }));
    }, 1000);
    return () => clearInterval(i);
  }, []);

  useEffect(() => {
    const i = setInterval(() => setShowBoxes(v => !v), 3500 + Math.random() * 2000);
    return () => clearInterval(i);
  }, []);

  return (
    <div className={clsx("cctv-wrapper rounded-xl border overflow-hidden", cfg.border)}>
      <div className="relative aspect-video bg-black">
        <video src={videoSrc} autoPlay muted loop playsInline
          className="w-full h-full object-cover"
          style={{ filter: "brightness(0.85) contrast(1.1) saturate(0.8)" }}
        />
        <div className="scanline" />
        <div className="cctv-corner cctv-corner-tl" />
        <div className="cctv-corner cctv-corner-tr" />
        <div className="cctv-corner cctv-corner-bl" />
        <div className="cctv-corner cctv-corner-br" />
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-black/30 pointer-events-none" />

        {/* Top bar */}
        <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-3 py-2 z-10">
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse-fast" />
            <span className="text-[9px] font-bold text-white font-mono">● REC</span>
          </div>
          <span className="text-[9px] text-cyan-300 font-mono bg-black/50 px-2 py-0.5 rounded">{time}</span>
        </div>
        <div className="absolute top-8 left-3 z-10">
          <span className="text-[9px] text-slate-300 font-mono bg-black/50 px-2 py-0.5 rounded">{cameraId}</span>
        </div>

        {/* Detection boxes */}
        {showBoxes && detections.map((det, i) => (
          <div key={i} className="detection-box"
            style={{ top: det.top, left: det.left, width: det.width, height: det.height, borderColor: det.color }}>
            <span className="absolute -top-4 left-0 text-[8px] font-bold font-mono px-1 py-0.5 rounded-sm whitespace-nowrap"
              style={{ background: det.color + "33", color: det.color, border: `1px solid ${det.color}60` }}>
              {det.label}
            </span>
          </div>
        ))}

        {/* Bottom bar */}
        <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between px-3 py-2 z-10"
          style={{ background: "linear-gradient(to top,rgba(0,0,0,0.85),transparent)" }}>
          <p className="text-[9px] text-slate-400 font-mono">{location}</p>
          <div className={clsx("flex items-center gap-1.5 px-2 py-0.5 rounded border text-[9px] font-bold font-mono", cfg.color, cfg.border, cfg.bg)}>
            <span className={clsx("w-1.5 h-1.5 rounded-full", status === "critical" ? "bg-red-400 animate-pulse-fast" : status === "warning" ? "bg-amber-400 animate-pulse" : "bg-green-400")} />
            {cfg.label}
          </div>
        </div>
      </div>
    </div>
  );
}
