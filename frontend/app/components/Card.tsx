import clsx from "clsx";
import { ReactNode } from "react";

interface CardProps {
  children: ReactNode;
  className?: string;
  glow?: "cyan" | "red" | "amber" | "emerald";
}

export function Card({ children, className, glow }: CardProps) {
  return (
    <div
      className={clsx(
        "rounded-2xl border border-slate-800 bg-slate-900/60 backdrop-blur-sm p-5",
        "transition-all duration-300 hover:border-slate-700",
        glow === "cyan"    && "hover:shadow-[0_0_24px_rgba(6,182,212,0.12)]",
        glow === "red"     && "hover:shadow-[0_0_24px_rgba(239,68,68,0.12)]",
        glow === "amber"   && "hover:shadow-[0_0_24px_rgba(245,158,11,0.12)]",
        glow === "emerald" && "hover:shadow-[0_0_24px_rgba(52,211,153,0.12)]",
        className
      )}
    >
      {children}
    </div>
  );
}

interface KpiCardProps {
  icon: ReactNode;
  label: string;
  value: string | number;
  sub?: string;
  color?: "cyan" | "emerald" | "amber" | "red" | "indigo";
}

const colorMap = {
  cyan:    "from-cyan-500/20 to-cyan-600/5 border-cyan-500/20 text-cyan-400",
  emerald: "from-emerald-500/20 to-emerald-600/5 border-emerald-500/20 text-emerald-400",
  amber:   "from-amber-500/20 to-amber-600/5 border-amber-500/20 text-amber-400",
  red:     "from-red-500/20 to-red-600/5 border-red-500/20 text-red-400",
  indigo:  "from-indigo-500/20 to-indigo-600/5 border-indigo-500/20 text-indigo-400",
};

export function KpiCard({ icon, label, value, sub, color = "cyan" }: KpiCardProps) {
  const cls = colorMap[color];
  return (
    <div className={clsx(
      "rounded-2xl border bg-gradient-to-br p-5 animate-fade-in",
      cls
    )}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">{label}</p>
          <p className="text-3xl font-bold text-white">{value}</p>
          {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
        </div>
        <div className={clsx("p-2.5 rounded-xl bg-slate-800/60", cls.split(" ").pop())}>
          {icon}
        </div>
      </div>
    </div>
  );
}

export function SectionTitle({ children }: { children: ReactNode }) {
  return (
    <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-4">
      {children}
    </h2>
  );
}

export function Spinner() {
  return (
    <div className="flex items-center justify-center h-40">
      <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
    </div>
  );
}

export function ErrorMsg({ msg }: { msg: string }) {
  return (
    <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 text-sm text-red-400">
      {msg}
    </div>
  );
}
