import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Urban Anomaly Detection · Bangalore",
  description: "ML-powered urban anomaly detection dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-slate-950 text-slate-200 antialiased">{children}</body>
    </html>
  );
}
