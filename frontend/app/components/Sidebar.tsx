"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard, BarChart3, ScatterChart, Zap,
  Camera, MapPin, Activity, Shield, Radio, Cpu,
} from "lucide-react";
import clsx from "clsx";

const NAV = [
  { href: "/dashboard",             icon: LayoutDashboard, label: "Command Center",  badge: "LIVE" },
  { href: "/dashboard/surveillance",icon: Camera,          label: "Surveillance",    badge: null },
  { href: "/dashboard/map",         icon: MapPin,          label: "City Map",        badge: null },
  { href: "/dashboard/analytics",   icon: BarChart3,       label: "Analytics",       badge: null },
  { href: "/dashboard/clustering",  icon: ScatterChart,    label: "AI Clustering",   badge: null },
  { href: "/dashboard/predictions", icon: Zap,             label: "Incident Analyzer", badge: null },
];

export default function Sidebar() {
  const path = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 flex flex-col z-40 border-r border-cyan-900/30"
      style={{ background: "rgba(2,6,23,0.97)", backdropFilter: "blur(20px)" }}>

      {/* Logo / Branding */}
      <div className="px-5 py-5 border-b border-cyan-900/30">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-700 flex items-center justify-center shadow-neon-cyan">
              <Shield size={18} className="text-white" />
            </div>
            <span className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full border-2 border-slate-950 animate-pulse" />
          </div>
          <div>
            <p className="text-xs font-black text-white tracking-wider uppercase" style={{ fontFamily: "'Orbitron', sans-serif" }}>
              BSCIP
            </p>
            <p className="text-[10px] text-cyan-500 leading-tight font-mono">Bangalore Smart City</p>
          </div>
        </div>

        {/* System status */}
        <div className="mt-4 p-2.5 rounded-lg bg-green-500/5 border border-green-500/20">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-[10px] text-green-400 font-semibold font-mono">SYSTEM OPERATIONAL</span>
          </div>
          <div className="flex items-center gap-3 mt-1.5">
            <span className="text-[9px] text-slate-600 font-mono">AI: ONLINE</span>
            <span className="text-[9px] text-slate-600 font-mono">ML: ACTIVE</span>
            <span className="text-[9px] text-slate-600 font-mono">API: OK</span>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        <p className="px-3 mb-3 text-[9px] font-bold text-slate-700 uppercase tracking-[0.2em] font-mono">
          Control Modules
        </p>
        {NAV.map(({ href, icon: Icon, label, badge }) => {
          const active = path === href || (href !== "/dashboard" && path.startsWith(href));
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "relative flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 group",
                active
                  ? "bg-cyan-500/10 text-cyan-300 border border-cyan-500/25 shadow-[0_0_12px_rgba(6,182,212,0.1)]"
                  : "text-slate-500 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent"
              )}
            >
              {active && (
                <span className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-6 bg-cyan-400 rounded-full shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
              )}
              <Icon size={15} className={active ? "text-cyan-400" : "text-slate-600 group-hover:text-slate-400"} />
              <span className="flex-1 text-xs">{label}</span>
              {badge && (
                <span className="text-[8px] font-bold font-mono bg-green-500/20 text-green-400 border border-green-500/30 px-1.5 py-0.5 rounded animate-pulse">
                  {badge}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* System Stats */}
      <div className="px-4 py-4 border-t border-cyan-900/30 space-y-3">
        <p className="text-[9px] font-bold text-slate-700 uppercase tracking-[0.2em] font-mono">Model Stack</p>
        <div className="space-y-1.5">
          {[
            { label: "Random Forest", color: "bg-cyan-500",   pct: "98.9%" },
            { label: "XGBoost",       color: "bg-indigo-500", pct: "98.8%" },
            { label: "DBSCAN",        color: "bg-purple-500", pct: "Active" },
            { label: "PCA (2D)",      color: "bg-blue-500",   pct: "45.0%" },
          ].map(m => (
            <div key={m.label} className="flex items-center gap-2">
              <span className={clsx("w-1.5 h-1.5 rounded-full flex-shrink-0", m.color)} />
              <span className="text-[9px] text-slate-600 flex-1 font-mono">{m.label}</span>
              <span className="text-[9px] text-slate-500 font-mono">{m.pct}</span>
            </div>
          ))}
        </div>
        <div className="flex items-center gap-2 pt-1">
          <Cpu size={10} className="text-cyan-700" />
          <span className="text-[9px] text-slate-700 font-mono">v2.4.1 · Local Mode</span>
        </div>
      </div>
    </aside>
  );
}
