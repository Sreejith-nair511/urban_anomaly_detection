import Sidebar from "../components/Sidebar";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen bg-slate-950">
      <Sidebar />
      <div className="flex-1 ml-60 flex flex-col min-h-screen">
        {/* Top navbar */}
        <header className="sticky top-0 z-30 h-14 flex items-center justify-between px-8 glass border-b border-slate-800">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500">Smart Urban Anomaly Detection</span>
            <span className="text-slate-700">/</span>
            <span className="text-xs text-slate-300 font-medium">Bangalore</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-[10px] text-slate-600 bg-slate-800 px-2 py-1 rounded-md">
              ML Pipeline Active
            </span>
            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-cyan-500 to-indigo-600 flex items-center justify-center text-[10px] font-bold text-white">
              AI
            </div>
          </div>
        </header>
        <main className="flex-1 p-8 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}
