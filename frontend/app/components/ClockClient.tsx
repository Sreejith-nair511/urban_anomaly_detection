"use client";
import { useEffect, useState } from "react";

export default function ClockClient() {
  const [time, setTime] = useState("");
  const [date, setDate] = useState("");

  useEffect(() => {
    const update = () => {
      const now = new Date();
      setTime(now.toLocaleTimeString("en-IN", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }));
      setDate(now.toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" }));
    };
    update();
    const i = setInterval(update, 1000);
    return () => clearInterval(i);
  }, []);

  return (
    <div className="text-right">
      <p className="text-[11px] font-bold font-mono text-slate-300 tracking-wider">{time}</p>
      <p className="text-[8px] text-slate-600 font-mono">{date} · IST</p>
    </div>
  );
}
