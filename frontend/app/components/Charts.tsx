"use client";
import {
  BarChart, Bar, LineChart, Line, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell, RadarChart, Radar,
  PolarGrid, PolarAngleAxis,
} from "recharts";

const COLORS = {
  Normal:        "#34d399",
  Moderate:      "#fbbf24",
  "High Anomaly":"#f87171",
  cyan:          "#06b6d4",
  indigo:        "#818cf8",
};

const TT_STYLE = {
  backgroundColor: "#0f172a",
  border: "1px solid rgba(148,163,184,0.1)",
  borderRadius: "10px",
  color: "#e2e8f0",
  fontSize: 12,
};

// ── AQI Histogram ────────────────────────────────────────────────────────────
export function AqiHistogram({ data }: { data: { range: string; count: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="range" tick={{ fontSize: 10 }} interval={1} />
        <YAxis tick={{ fontSize: 10 }} />
        <Tooltip contentStyle={TT_STYLE} />
        <Bar dataKey="count" radius={[4, 4, 0, 0]}>
          {data.map((_, i) => (
            <Cell
              key={i}
              fill={i < 3 ? COLORS.Normal : i < 6 ? COLORS.Moderate : COLORS["High Anomaly"]}
              fillOpacity={0.85}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Hourly Trend ─────────────────────────────────────────────────────────────
export function HourlyTrend({
  data,
}: {
  data: { hour: number; AQI: number; traffic_density: number; noise_level: number }[];
}) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="hour" tick={{ fontSize: 10 }} label={{ value: "Hour", position: "insideBottom", offset: -2, fontSize: 10, fill: "#64748b" }} />
        <YAxis tick={{ fontSize: 10 }} />
        <Tooltip contentStyle={TT_STYLE} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Line type="monotone" dataKey="AQI"             stroke={COLORS["High Anomaly"]} strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="traffic_density" stroke={COLORS.cyan}            strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="noise_level"     stroke={COLORS.indigo}          strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ── Zone Anomaly Bar ──────────────────────────────────────────────────────────
export function ZoneAnomalyBar({ data }: { data: { zone: string; anomaly_rate: number }[] }) {
  const sorted = [...data].sort((a, b) => b.anomaly_rate - a.anomaly_rate);
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={sorted} layout="vertical" margin={{ top: 4, right: 16, left: 60, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" tick={{ fontSize: 10 }} unit="%" />
        <YAxis type="category" dataKey="zone" tick={{ fontSize: 10 }} width={80} />
        <Tooltip contentStyle={TT_STYLE} formatter={(v: number) => [`${v}%`, "Anomaly Rate"]} />
        <Bar dataKey="anomaly_rate" radius={[0, 4, 4, 0]}>
          {sorted.map((d, i) => (
            <Cell
              key={i}
              fill={d.anomaly_rate > 3 ? COLORS["High Anomaly"] : d.anomaly_rate > 1.5 ? COLORS.Moderate : COLORS.Normal}
              fillOpacity={0.85}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── PCA Scatter ───────────────────────────────────────────────────────────────
export function PcaScatter({
  data,
}: {
  data: { x: number; y: number; label: string; zone: string }[];
}) {
  const groups = ["Normal", "Moderate", "High Anomaly"] as const;
  return (
    <ResponsiveContainer width="100%" height={340}>
      <ScatterChart margin={{ top: 8, right: 8, left: -10, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="x" type="number" name="PC1" tick={{ fontSize: 10 }} />
        <YAxis dataKey="y" type="number" name="PC2" tick={{ fontSize: 10 }} />
        <Tooltip
          contentStyle={TT_STYLE}
          cursor={{ strokeDasharray: "3 3" }}
          content={({ payload }) => {
            if (!payload?.length) return null;
            const d = payload[0].payload;
            return (
              <div style={TT_STYLE} className="p-2 text-xs">
                <p className="font-semibold">{d.label}</p>
                <p className="text-slate-400">{d.zone}</p>
                <p>PC1: {d.x.toFixed(3)}</p>
                <p>PC2: {d.y.toFixed(3)}</p>
              </div>
            );
          }}
        />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {groups.map((g) => (
          <Scatter
            key={g}
            name={g}
            data={data.filter((d) => d.label === g)}
            fill={COLORS[g]}
            fillOpacity={g === "High Anomaly" ? 0.9 : 0.5}
            shape={g === "High Anomaly" ? "cross" : "circle"}
          />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

// ── Feature Importance Bar ────────────────────────────────────────────────────
export function FeatureImportanceBar({
  data,
  color = COLORS.cyan,
}: {
  data: { feature: string; importance: number }[];
  color?: string;
}) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart
        data={[...data].reverse()}
        layout="vertical"
        margin={{ top: 4, right: 16, left: 80, bottom: 0 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" tick={{ fontSize: 10 }} tickFormatter={(v) => v.toFixed(2)} />
        <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={90} />
        <Tooltip contentStyle={TT_STYLE} formatter={(v: number) => [v.toFixed(4), "Importance"]} />
        <Bar dataKey="importance" fill={color} fillOpacity={0.85} radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Confidence Radar ──────────────────────────────────────────────────────────
export function ConfidenceRadar({
  data,
}: {
  data: { label: string; rf: number; xgb: number }[];
}) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <RadarChart data={data}>
        <PolarGrid stroke="rgba(148,163,184,0.1)" />
        <PolarAngleAxis dataKey="label" tick={{ fontSize: 11, fill: "#94a3b8" }} />
        <Radar name="Random Forest" dataKey="rf"  stroke={COLORS.cyan}   fill={COLORS.cyan}   fillOpacity={0.2} />
        <Radar name="XGBoost"       dataKey="xgb" stroke={COLORS.indigo} fill={COLORS.indigo} fillOpacity={0.2} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Tooltip contentStyle={TT_STYLE} formatter={(v: number) => [`${(v * 100).toFixed(1)}%`]} />
      </RadarChart>
    </ResponsiveContainer>
  );
}
