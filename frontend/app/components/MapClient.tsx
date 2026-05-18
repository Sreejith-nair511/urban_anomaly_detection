"use client";
import { useEffect, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { MapPin } from "lucide-react";

const ZONES = [
  { id: "KOR", name: "Koramangala",    lat: 12.9352, lng: 77.6245, aqi: 312, traffic: 89, noise: 97, anomaly: "critical" },
  { id: "WHF", name: "Whitefield",     lat: 12.9698, lng: 77.7499, aqi: 387, traffic: 72, noise: 68, anomaly: "critical" },
  { id: "IND", name: "Indiranagar",    lat: 12.9784, lng: 77.6408, aqi: 145, traffic: 55, noise: 72, anomaly: "warning"  },
  { id: "HEB", name: "Hebbal",         lat: 13.0358, lng: 77.5970, aqi: 178, traffic: 91, noise: 85, anomaly: "critical" },
  { id: "ELC", name: "Electronic City",lat: 12.8458, lng: 77.6692, aqi: 198, traffic: 63, noise: 74, anomaly: "warning"  },
  { id: "JAY", name: "Jayanagar",      lat: 12.9250, lng: 77.5938, aqi:  72, traffic: 28, noise: 44, anomaly: "normal"   },
  { id: "MAR", name: "Marathahalli",   lat: 12.9591, lng: 77.7001, aqi: 225, traffic: 94, noise: 91, anomaly: "critical" },
  { id: "YEL", name: "Yelahanka",      lat: 13.1007, lng: 77.5963, aqi:  88, traffic: 31, noise: 52, anomaly: "normal"   },
  { id: "BTM", name: "BTM Layout",     lat: 12.9166, lng: 77.6101, aqi: 195, traffic: 67, noise: 78, anomaly: "warning"  },
  { id: "HSR", name: "HSR Layout",     lat: 12.9116, lng: 77.6389, aqi: 162, traffic: 58, noise: 104, anomaly: "warning" },
];

const SEVERITY = {
  critical: { color: "#ef4444", label: "HIGH ANOMALY" },
  warning:  { color: "#f59e0b", label: "MODERATE ACTIVITY" },
  normal:   { color: "#10b981", label: "NORMAL" },
};

function MapUpdater({ selected }: { selected: string | null }) {
  const map = useMap();
  useEffect(() => {
    if (selected) {
      const z = ZONES.find(x => x.id === selected);
      if (z) {
        map.setView([z.lat, z.lng], 14, { animate: true, duration: 1.5 });
      }
    } else {
      map.setView([12.9716, 77.5946], 11, { animate: true, duration: 1.5 });
    }
  }, [selected, map]);
  return null;
}

export default function MapClient({ onZoneClick }: { onZoneClick?: (zone: typeof ZONES[0]) => void }) {
  const [selected, setSelected] = useState<string | null>(null);

  const handleClick = (zone: typeof ZONES[0]) => {
    setSelected(zone.id);
    onZoneClick?.(zone);
  };

  const selectedZone = ZONES.find(z => z.id === selected);

  return (
    <div className="glass-card rounded-2xl p-4 h-full flex flex-col relative z-0">
      <div className="flex items-center justify-between mb-3 z-10 relative">
        <div className="flex items-center gap-2">
          <span className="w-0.5 h-4 bg-cyan-400 rounded-full shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
          <p className="text-[10px] font-bold text-slate-300 font-mono tracking-widest">BANGALORE ZONE MAP</p>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-[9px] font-mono">
          {Object.entries(SEVERITY).map(([key, val]) => (
            <div key={key} className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full" style={{ background: val.color }} />
              <span className="text-slate-300">{key.toUpperCase()}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="flex-1 rounded-xl overflow-hidden border border-cyan-900/20 relative isolate">
        <MapContainer
          center={[12.9716, 77.5946]}
          zoom={11}
          style={{ height: "100%", width: "100%", background: "#0a1628" }}
          zoomControl={false}
          attributionControl={false}
        >
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution="&copy; <a href='https://carto.com/attributions'>CARTO</a>"
          />
          <MapUpdater selected={selected} />

          {ZONES.map((zone) => {
            const sev = SEVERITY[zone.anomaly as keyof typeof SEVERITY];
            const isSelected = selected === zone.id;
            
            return (
              <CircleMarker
                key={zone.id}
                center={[zone.lat, zone.lng]}
                radius={isSelected ? 16 : 8}
                pathOptions={{
                  color: sev.color,
                  fillColor: sev.color,
                  fillOpacity: isSelected ? 0.6 : 0.3,
                  weight: isSelected ? 3 : 2,
                }}
                eventHandlers={{ click: () => handleClick(zone) }}
              >
                <Popup className="custom-popup" closeButton={false}>
                  <div className="text-center">
                    <p className="text-[12px] font-bold text-white mb-1" style={{ fontFamily: "monospace" }}>{zone.name}</p>
                    <span className="text-[9px] font-bold px-1.5 py-0.5 rounded" style={{ color: sev.color, border: `1px solid ${sev.color}60`, background: `${sev.color}20` }}>
                      {sev.label}
                    </span>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}
        </MapContainer>
      </div>

      {selectedZone && (() => {
        const sev = SEVERITY[selectedZone.anomaly as keyof typeof SEVERITY];
        return (
          <div className="mt-3 p-3 rounded-xl border relative z-10" style={{ borderColor: sev.color + "40", background: sev.color + "08" }}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <MapPin size={11} style={{ color: sev.color }} />
                <span className="text-xs font-bold text-white">{selectedZone.name}</span>
                <span className="text-[9px] font-bold font-mono px-1.5 py-0.5 rounded"
                  style={{ color: sev.color, background: sev.color + "20", border: `1px solid ${sev.color}40` }}>
                  {sev.label}
                </span>
              </div>
              <button onClick={() => { setSelected(null); onZoneClick?.(null as any); }} className="text-[9px] text-slate-500 hover:text-white font-mono underline">
                RESET VIEW
              </button>
            </div>
            <div className="grid grid-cols-4 gap-2 text-[10px]">
              {[
                { label: "AQI", val: selectedZone.aqi, unit: "", warn: 200, crit: 300 },
                { label: "Traffic", val: selectedZone.traffic, unit: "%", warn: 70, crit: 85 },
                { label: "Noise", val: selectedZone.noise, unit: "dB", warn: 75, crit: 90 },
              ].map(m => {
                const c = m.val >= m.crit ? "text-red-400" : m.val >= m.warn ? "text-amber-400" : "text-green-400";
                return (
                  <div key={m.label} className="text-center bg-slate-900/50 rounded py-1 border border-slate-700/30">
                    <p className="text-slate-500 font-mono text-[8px]">{m.label}</p>
                    <p className={`font-black font-mono ${c}`}>{m.val}{m.unit}</p>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })()}
    </div>
  );
}
