"use client";
import BangaloreMap from "../../components/BangaloreMap";
import { Card, SectionTitle } from "../../components/Card";
import { ZoneAnomalyBar } from "../../components/Charts";
import { MapPin, Activity } from "lucide-react";
import { useEffect, useState } from "react";
import { fetchSummary, DataSummary } from "../../lib/api";

const ZONE_DETAIL = [
  { id:"KOR", name:"Koramangala",     aqi:312, traffic:89, noise:97,  temp:38, anomaly:"critical" },
  { id:"WHF", name:"Whitefield",      aqi:387, traffic:72, noise:68,  temp:37, anomaly:"critical" },
  { id:"IND", name:"Indiranagar",     aqi:145, traffic:55, noise:72,  temp:32, anomaly:"warning"  },
  { id:"HEB", name:"Hebbal",          aqi:178, traffic:91, noise:85,  temp:33, anomaly:"critical" },
  { id:"ELC", name:"Electronic City", aqi:198, traffic:63, noise:74,  temp:39, anomaly:"warning"  },
  { id:"JAY", name:"Jayanagar",       aqi: 72, traffic:28, noise:44,  temp:29, anomaly:"normal"   },
  { id:"MAR", name:"Marathahalli",    aqi:225, traffic:94, noise:91,  temp:35, anomaly:"critical" },
  { id:"YEL", name:"Yelahanka",       aqi: 88, traffic:31, noise:52,  temp:28, anomaly:"normal"   },
  { id:"BTM", name:"BTM Layout",      aqi:195, traffic:67, noise:78,  temp:34, anomaly:"warning"  },
  { id:"HSR", name:"HSR Layout",      aqi:162, traffic:58, noise:104, temp:31, anomaly:"warning"  },
];

const SEV_COLOR = { critical:"text-red-400", warning:"text-amber-400", normal:"text-green-400" };

export default function MapPage() {
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [selected, setSelected] = useState<typeof ZONE_DETAIL[0] | null>(null);

  useEffect(() => {
    fetchSummary().then(r => setSummary(r.data)).catch(() => {});
  }, []);

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center gap-3">
        <MapPin size={20} className="text-cyan-400" />
        <h1 className="text-xl font-black text-white" style={{ fontFamily: "'Orbitron', sans-serif" }}>CITY MAP</h1>
        <span className="text-[9px] font-bold font-mono bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 px-2 py-1 rounded">10 ZONES MONITORED</span>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        {/* Map */}
        <div className="xl:col-span-2" style={{ minHeight: "500px" }}>
          <BangaloreMap onZoneClick={(z: any) => setSelected(ZONE_DETAIL.find(d => d.name === z.name) ?? null)} />
        </div>

        {/* Zone Detail Panel */}
        <div className="space-y-3">
          <Card>
            <SectionTitle>ZONE SENSOR STATUS</SectionTitle>
            <div className="space-y-2 overflow-y-auto max-h-96">
              {ZONE_DETAIL.map(z => {
                const col = SEV_COLOR[z.anomaly as keyof typeof SEV_COLOR];
                return (
                  <div key={z.id}
                    onClick={() => setSelected(z)}
                    className={`p-2.5 rounded-lg border cursor-pointer transition-all hover:scale-[1.01] ${
                      z.anomaly === "critical" ? "border-red-500/20 bg-red-500/5" :
                      z.anomaly === "warning"  ? "border-amber-500/20 bg-amber-500/5" :
                      "border-green-500/20 bg-green-500/5"
                    } ${selected?.id === z.id ? "ring-1 ring-cyan-500/40" : ""}`}>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-[10px] font-bold text-slate-300 font-mono">{z.name}</span>
                      <span className={`text-[8px] font-bold font-mono ${col}`}>{z.anomaly.toUpperCase()}</span>
                    </div>
                    <div className="grid grid-cols-4 gap-1 text-[9px] font-mono">
                      <div><p className="text-slate-700">AQI</p><p className={z.aqi>300?"text-red-400":z.aqi>200?"text-amber-400":"text-green-400"}>{z.aqi}</p></div>
                      <div><p className="text-slate-700">TRF</p><p className={z.traffic>80?"text-red-400":z.traffic>60?"text-amber-400":"text-green-400"}>{z.traffic}%</p></div>
                      <div><p className="text-slate-700">dB</p><p className={z.noise>90?"text-red-400":z.noise>70?"text-amber-400":"text-green-400"}>{z.noise}</p></div>
                      <div><p className="text-slate-700">°C</p><p className={z.temp>37?"text-red-400":z.temp>33?"text-amber-400":"text-green-400"}>{z.temp}</p></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>
        </div>
      </div>

      {/* Zone anomaly bar chart */}
      {summary && (
        <Card glow="cyan">
          <SectionTitle>ZONE ANOMALY RATES (%)</SectionTitle>
          <ZoneAnomalyBar data={summary.zone_anomaly_rates} />
        </Card>
      )}
    </div>
  );
}
