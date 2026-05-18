import Sidebar from "../components/Sidebar";
import Ticker from "../components/Ticker";
import ClockClient from "../components/ClockClient";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="h-screen bg-slate-950 bg-grid">
      <Sidebar />
      <div className="ml-64 flex flex-col h-screen w-[calc(100%-16rem)] overflow-hidden">
        {/* Emergency Alert Ticker */}
        <Ticker />

        {/* Top navbar */}
        <header
          className="sticky top-8 z-30 h-12 flex items-center justify-between px-6 border-b border-cyan-900/30"
          style={{ background: "rgba(2,6,23,0.95)", backdropFilter: "blur(20px)" }}
        >
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              <span className="text-[10px] text-green-400 font-mono font-bold tracking-wider">SYSTEM ONLINE</span>
            </div>
            <span className="text-slate-800">|</span>
            <span className="text-[10px] text-slate-600 font-mono">
              Bangalore Smart City Intelligence Platform
            </span>
          </div>

          <div className="flex items-center gap-3">
            <ClockClient />
            <div className="h-4 w-px bg-slate-800" />
            <div className="flex items-center gap-2 bg-cyan-500/5 border border-cyan-500/20 px-3 py-1 rounded-lg">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
              <span className="text-[10px] text-cyan-400 font-mono font-bold">AI PIPELINE ACTIVE</span>
            </div>
            <div
              className="w-7 h-7 rounded-full bg-gradient-to-br from-cyan-500 to-indigo-600 flex items-center justify-center text-[9px] font-black text-white shadow-neon-cyan"
              style={{ fontFamily: "'Orbitron', sans-serif" }}
            >
              AI
            </div>
          </div>
        </header>

        <main className="flex-1 p-6 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}
