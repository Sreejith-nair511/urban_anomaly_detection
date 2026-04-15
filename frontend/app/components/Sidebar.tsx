"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard, BarChart3, ScatterChart, Zap, Activity,
} from "lucide-react";
import clsx from "clsx";

const NAV = [
  { href: "/dashboard",             icon: LayoutDashboard, label: "Overview"    },
  { href: "/dashboard/analytics",   icon: BarChart3,       label: "Analytics"   },
  { href: "/dashboard/clustering",  icon: ScatterChart,    label: "Clustering"  },
  { href: "/dashboard/predictions", icon: Zap,             label: "Predictions" },
];

export default function Sidebar() {
  const path = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-screen w-60 flex flex-col glass border-r border-slate-800 z-40">
      {/* Logo */}
      <div className="px-6 py-6 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-indigo-600 flex items-center justify-center">
            <Activity size={16} className="text-white" />
          </div>
          <div>
            <p className="text-sm font-bold text-white leading-tight">AnomalyAI</p>
            <p className="text-[10px] text-slate-500 leading-tight">Bangalore · Urban</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        <p className="px-3 mb-3 text-[10px] font-semibold text-slate-600 uppercase tracking-widest">
          Navigation
        </p>
        {NAV.map(({ href, icon: Icon, label }) => {
          const active = path === href || (href !== "/dashboard" && path.startsWith(href));
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200",
                active
                  ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/20"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              )}
            >
              <Icon size={16} className={active ? "text-cyan-400" : "text-slate-500"} />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-4 py-4 border-t border-slate-800">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse-slow" />
          <span className="text-xs text-slate-500">API Connected</span>
        </div>
        <p className="text-[10px] text-slate-700 mt-1">RF · XGBoost · DBSCAN · PCA</p>
      </div>
    </aside>
  );
}
