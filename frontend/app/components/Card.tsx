"use client";
import clsx from "clsx";
import { ReactNode, useEffect, useRef, useState } from "react";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

// ── Animated counter hook ─────────────────────────────────────────────────────
function useCounter(target: number, duration = 1200) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    let start = 0;
    const step = target / (duration / 16);
    const timer = setInterval(() => {
      start += step;
      if (start >= target) { setValue(target); clearInterval(timer); }
      else setValue(Math.floor(start));
    }, 16);
    return () => clearInterval(timer);
  }, [target, duration]);
  return value;
}

// ── Base Card ─────────────────────────────────────────────────────────────────
interface CardProps { children: ReactNode; className?: string; glow?: "cyan"|"red"|"amber"|"emerald"|"purple"; }
export function Card({ children, className, glow }: CardProps) {
  return (
    <div className={clsx(
      "glass-card rounded-2xl p-5 transition-all duration-300",
      glow === "cyan"    && "hover:border-cyan-500/40   hover:shadow-[0_0_24px_rgba(6,182,212,0.12)]",
      glow === "red"     && "hover:border-red-500/40    hover:shadow-[0_0_24px_rgba(239,68,68,0.12)]",
      glow === "amber"   && "hover:border-amber-500/40  hover:shadow-[0_0_24px_rgba(245,158,11,0.12)]",
      glow === "emerald" && "hover:border-emerald-500/40 hover:shadow-[0_0_24px_rgba(16,185,129,0.12)]",
      glow === "purple"  && "hover:border-purple-500/40 hover:shadow-[0_0_24px_rgba(139,92,246,0.12)]",
      className
    )}>
      {children}
    </div>
  );
}

// ── KPI Card with animated counter ───────────────────────────────────────────
type KpiColor = "cyan"|"emerald"|"amber"|"red"|"indigo"|"purple";

const colorCfg: Record<KpiColor, {
  ring: string; icon: string; text: string; bg: string; border: string; glow: string;
}> = {
  cyan:    { ring: "from-cyan-500   to-blue-600",   icon: "text-cyan-400",    text: "text-cyan-400",    bg: "bg-cyan-500/8",    border: "border-cyan-500/20",   glow: "shadow-[0_0_20px_rgba(6,182,212,0.15)]"   },
  emerald: { ring: "from-emerald-500 to-green-600", icon: "text-emerald-400", text: "text-emerald-400", bg: "bg-emerald-500/8", border: "border-emerald-500/20", glow: "shadow-[0_0_20px_rgba(16,185,129,0.15)]"  },
  amber:   { ring: "from-amber-500  to-orange-600", icon: "text-amber-400",   text: "text-amber-400",   bg: "bg-amber-500/8",   border: "border-amber-500/20",   glow: "shadow-[0_0_20px_rgba(245,158,11,0.15)]"  },
  red:     { ring: "from-red-500    to-rose-600",   icon: "text-red-400",     text: "text-red-400",     bg: "bg-red-500/8",     border: "border-red-500/20",     glow: "shadow-[0_0_20px_rgba(239,68,68,0.15)]"   },
  indigo:  { ring: "from-indigo-500 to-violet-600", icon: "text-indigo-400",  text: "text-indigo-400",  bg: "bg-indigo-500/8",  border: "border-indigo-500/20",  glow: "shadow-[0_0_20px_rgba(129,140,248,0.15)]" },
  purple:  { ring: "from-purple-500 to-fuchsia-600",icon: "text-purple-400",  text: "text-purple-400",  bg: "bg-purple-500/8",  border: "border-purple-500/20",  glow: "shadow-[0_0_20px_rgba(139,92,246,0.15)]"  },
};

interface KpiCardProps {
  icon: ReactNode; label: string; value: number | string; unit?: string;
  sub?: string; trend?: "up"|"down"|"flat"; color?: KpiColor; animate?: boolean;
}
export function KpiCard({ icon, label, value, unit = "", sub, trend, color = "cyan", animate = true }: KpiCardProps) {
  const cfg = colorCfg[color];
  const numericValue = typeof value === "number" ? value : parseFloat(String(value));
  const isNumeric = !isNaN(numericValue) && animate;
  const counted = useCounter(isNumeric ? Math.floor(numericValue) : 0);
  const displayVal = isNumeric ? counted : value;

  const TrendIcon = trend === "up" ? TrendingUp : trend === "down" ? TrendingDown : Minus;
  const trendColor = trend === "up" ? "text-red-400" : trend === "down" ? "text-green-400" : "text-slate-500";

  return (
    <div className={clsx(
      "relative rounded-2xl p-5 border overflow-hidden transition-all duration-300 group cursor-default",
      cfg.bg, cfg.border, cfg.glow,
      "hover:scale-[1.02]"
    )}>
      {/* Background gradient orb */}
      <div className={clsx(
        "absolute -right-4 -top-4 w-20 h-20 rounded-full bg-gradient-to-br opacity-10 blur-xl group-hover:opacity-20 transition-opacity",
        cfg.ring
      )} />

      <div className="relative flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-[0.15em] mb-2 font-mono">
            {label}
          </p>
          <div className="flex items-end gap-1">
            <p className={clsx("text-3xl font-black leading-none", cfg.text)} style={{ fontFamily: "'Orbitron', sans-serif" }}>
              {displayVal}
            </p>
            {unit && <span className="text-sm text-slate-500 mb-0.5">{unit}</span>}
          </div>
          {sub && <p className="text-[10px] text-slate-600 mt-1.5 font-mono">{sub}</p>}
        </div>
        <div className="flex flex-col items-end gap-2">
          <div className={clsx("p-2.5 rounded-xl bg-slate-800/60 border border-slate-700/40", cfg.icon)}>
            {icon}
          </div>
          {trend && (
            <div className={clsx("flex items-center gap-1", trendColor)}>
              <TrendIcon size={10} />
              <span className="text-[9px] font-mono">{trend === "up" ? "HIGH" : trend === "down" ? "LOW" : "STABLE"}</span>
            </div>
          )}
        </div>
      </div>

      {/* Bottom accent line */}
      <div className={clsx("absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r opacity-60", cfg.ring)} />
    </div>
  );
}

// ── Section Title ─────────────────────────────────────────────────────────────
export function SectionTitle({ children, accent = true }: { children: ReactNode; accent?: boolean }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      {accent && <span className="w-0.5 h-4 bg-cyan-400 rounded-full shadow-[0_0_8px_rgba(6,182,212,0.8)]" />}
      <h2 className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] font-mono">{children}</h2>
    </div>
  );
}

// ── Spinner ───────────────────────────────────────────────────────────────────
export function Spinner() {
  return (
    <div className="flex flex-col items-center justify-center h-60 gap-4">
      <div className="relative w-12 h-12">
        <div className="absolute inset-0 border-2 border-cyan-500/20 rounded-full" />
        <div className="absolute inset-0 border-2 border-t-cyan-400 border-r-transparent border-b-transparent border-l-transparent rounded-full animate-spin" />
        <div className="absolute inset-2 border-2 border-t-transparent border-r-indigo-400 border-b-transparent border-l-transparent rounded-full animate-spin" style={{ animationDirection: "reverse", animationDuration: "0.8s" }} />
      </div>
      <p className="text-[10px] text-cyan-600 font-mono tracking-widest animate-pulse">LOADING INTEL...</p>
    </div>
  );
}

// ── ErrorMsg ──────────────────────────────────────────────────────────────────
export function ErrorMsg({ msg }: { msg: string }) {
  return (
    <div className="rounded-xl border border-red-500/30 bg-red-500/5 p-5 text-sm text-red-400 font-mono">
      <p className="font-bold mb-1">⚠ CONNECTION ERROR</p>
      <p className="text-xs text-red-500/70">{msg}</p>
    </div>
  );
}

// ── Progress Bar ──────────────────────────────────────────────────────────────
export function ProgressBar({ value, max, color = "cyan" }: { value: number; max: number; color?: string }) {
  const pct = Math.min((value / max) * 100, 100);
  const colorMap: Record<string, string> = {
    cyan:    "bg-cyan-500",
    red:     "bg-red-500",
    amber:   "bg-amber-500",
    emerald: "bg-emerald-500",
    indigo:  "bg-indigo-500",
  };
  return (
    <div className="h-1.5 bg-slate-800/60 rounded-full overflow-hidden">
      <div
        className={clsx("h-full rounded-full transition-all duration-700", colorMap[color] ?? "bg-cyan-500")}
        style={{ width: `${pct}%`, boxShadow: `0 0 6px currentColor` }}
      />
    </div>
  );
}
