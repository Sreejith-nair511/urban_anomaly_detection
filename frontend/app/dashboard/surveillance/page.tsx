"use client";
import { useState } from "react";
import { SectionTitle, Card } from "../../components/Card";
import CCTVPanel from "../../components/CCTVPanel";
import { Camera, AlertTriangle, CheckCircle, Radio } from "lucide-react";

const ALL_FEEDS = [
  { videoSrc: "/videos/v1.mp4", cameraId: "CAM-01 · SILK BOARD",    location: "Silk Board Junction, ORR",        status: "critical" as const,
    detections: [{ label:"HIGH TRAFFIC", top:"20%",left:"10%",width:"35%",height:"50%",color:"#ef4444" }, { label:"CONGESTION",top:"40%",left:"55%",width:"30%",height:"35%",color:"#f59e0b" }] },
  { videoSrc: "/videos/v2.mp4", cameraId: "CAM-02 · WHITEFIELD",    location: "Whitefield Main Rd, ITPL Gate",   status: "warning"  as const,
    detections: [{ label:"POLLUTION SPIKE",top:"15%",left:"25%",width:"40%",height:"30%",color:"#f59e0b" }] },
  { videoSrc: "/videos/v3.mp4", cameraId: "CAM-03 · MARATHAHALLI",  location: "Marathahalli Bridge, ORR",        status: "critical" as const,
    detections: [{ label:"COMBINED ANOMALY",top:"20%",left:"15%",width:"70%",height:"60%",color:"#ef4444" }] },
  { videoSrc: "/videos/v4.mp4", cameraId: "CAM-04 · HEBBAL",        location: "Hebbal Flyover, NH-44",           status: "critical" as const,
    detections: [{ label:"TRAFFIC SURGE",top:"30%",left:"20%",width:"50%",height:"40%",color:"#ef4444" }] },
  { videoSrc: "/videos/v5.mp4", cameraId: "CAM-05 · INDIRANAGAR",   location: "Indiranagar 100ft Rd",           status: "warning"  as const,
    detections: [{ label:"MODERATE FLOW",top:"40%",left:"30%",width:"40%",height:"30%",color:"#f59e0b" }] },
  { videoSrc: "/videos/v6.mp4", cameraId: "CAM-06 · JAYANAGAR",     location: "Jayanagar 4th Block Circle",     status: "normal"   as const,
    detections: [] },
  { videoSrc: "/videos/v7.mp4", cameraId: "CAM-07 · ELECTRONIC CITY", location: "Electronic City Phase 2",     status: "warning"  as const,
    detections: [{ label:"THERMAL ANOMALY",top:"25%",left:"35%",width:"40%",height:"40%",color:"#f59e0b" }] },
];

const STATS = [
  { label: "Total Cameras",    value: "7",  color: "text-cyan-400"  },
  { label: "Critical",         value: "3",  color: "text-red-400"   },
  { label: "Warning",          value: "3",  color: "text-amber-400" },
  { label: "Normal",           value: "1",  color: "text-green-400" },
];

export default function SurveillancePage() {
  const [filter, setFilter] = useState<"all"|"critical"|"warning"|"normal">("all");
  const filtered = filter === "all" ? ALL_FEEDS : ALL_FEEDS.filter(f => f.status === filter);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <Camera size={20} className="text-cyan-400" />
            <h1 className="text-xl font-black text-white" style={{ fontFamily: "'Orbitron', sans-serif" }}>SURVEILLANCE</h1>
            <span className="text-[9px] font-bold font-mono bg-red-500/10 text-red-400 border border-red-500/30 px-2 py-1 rounded animate-pulse">● 7 FEEDS LIVE</span>
          </div>
          <p className="text-[11px] text-slate-500 font-mono ml-8">AI-powered CCTV monitoring across Bangalore junctions</p>
        </div>
        <div className="flex gap-2">
          {(["all","critical","warning","normal"] as const).map(f => (
            <button key={f} onClick={() => setFilter(f)}
              className={`text-[9px] font-bold font-mono px-3 py-1.5 rounded-lg border transition-all ${
                filter === f
                  ? f === "critical" ? "bg-red-500/20 text-red-400 border-red-500/40"
                  : f === "warning"  ? "bg-amber-500/20 text-amber-400 border-amber-500/40"
                  : f === "normal"   ? "bg-green-500/20 text-green-400 border-green-500/40"
                  : "bg-cyan-500/20 text-cyan-400 border-cyan-500/40"
                  : "bg-slate-800/40 text-slate-500 border-slate-700/40 hover:text-slate-300"
              }`}>
              {f.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-4 gap-3">
        {STATS.map(s => (
          <div key={s.label} className="glass-card rounded-xl p-3 text-center">
            <p className={`text-2xl font-black font-mono ${s.color}`} style={{ fontFamily: "'Orbitron',sans-serif" }}>{s.value}</p>
            <p className="text-[9px] text-slate-600 font-mono mt-1">{s.label}</p>
          </div>
        ))}
      </div>

      {/* CCTV Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {filtered.map((feed, i) => <CCTVPanel key={i} {...feed} />)}
      </div>
    </div>
  );
}
