"use client";
import PredictionForm from "../../components/PredictionForm";

export default function PredictionsPage() {
  return (
    <div className="space-y-8 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-white">Anomaly Predictor</h1>
        <p className="text-sm text-slate-500 mt-1">
          Input live sensor readings to get instant RF + XGBoost predictions
        </p>
      </div>

      <PredictionForm />

      {/* Info cards */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        {[
          {
            title: "Normal",
            desc:  "AQI < 150, traffic < 70, noise < 80. Typical urban conditions.",
            color: "emerald",
            emoji: "✅",
          },
          {
            title: "Moderate",
            desc:  "Elevated readings in one or more sensors. Monitor closely.",
            color: "amber",
            emoji: "⚠️",
          },
          {
            title: "High Anomaly",
            desc:  "AQI > 300, traffic > 80, or noise > 80. Immediate attention required.",
            color: "red",
            emoji: "🚨",
          },
        ].map(({ title, desc, color, emoji }) => (
          <div
            key={title}
            className={`rounded-2xl border p-4 bg-${color}-500/5 border-${color}-500/20`}
          >
            <p className="text-lg mb-1">{emoji}</p>
            <p className={`text-sm font-semibold text-${color}-400 mb-1`}>{title}</p>
            <p className="text-xs text-slate-500">{desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
