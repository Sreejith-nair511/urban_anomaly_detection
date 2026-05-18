/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        slate: { 950: "#020617" },
        neon: {
          cyan:   "#06b6d4",
          green:  "#10b981",
          amber:  "#f59e0b",
          red:    "#ef4444",
          purple: "#8b5cf6",
          blue:   "#3b82f6",
        },
      },
      backgroundImage: {
        "gradient-radial":  "radial-gradient(var(--tw-gradient-stops))",
        "grid-pattern":     "linear-gradient(rgba(6,182,212,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(6,182,212,0.03) 1px, transparent 1px)",
        "scan-gradient":    "linear-gradient(180deg, transparent 0%, rgba(6,182,212,0.05) 50%, transparent 100%)",
      },
      backgroundSize: {
        "grid-sm": "40px 40px",
      },
      boxShadow: {
        "neon-cyan":   "0 0 20px rgba(6,182,212,0.4), 0 0 40px rgba(6,182,212,0.1)",
        "neon-red":    "0 0 20px rgba(239,68,68,0.4),  0 0 40px rgba(239,68,68,0.1)",
        "neon-amber":  "0 0 20px rgba(245,158,11,0.4), 0 0 40px rgba(245,158,11,0.1)",
        "neon-green":  "0 0 20px rgba(16,185,129,0.4), 0 0 40px rgba(16,185,129,0.1)",
        "card-glow":   "0 4px 32px rgba(6,182,212,0.08), inset 0 1px 0 rgba(255,255,255,0.05)",
      },
      animation: {
        "pulse-slow":    "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "pulse-fast":    "pulse 1s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-in":       "fadeIn 0.4s ease-out",
        "slide-up":      "slideUp 0.5s ease-out",
        "ticker":        "ticker 40s linear infinite",
        "scanner":       "scanner 3s ease-in-out infinite",
        "glow-pulse":    "glowPulse 2s ease-in-out infinite",
        "float":         "float 4s ease-in-out infinite",
        "blink":         "blink 1.2s step-end infinite",
        "counter-up":    "counterUp 0.6s ease-out",
        "spin-slow":     "spin 8s linear infinite",
      },
      keyframes: {
        fadeIn:    { "0%": { opacity: "0", transform: "translateY(8px)"  }, "100%": { opacity: "1", transform: "translateY(0)" } },
        slideUp:   { "0%": { opacity: "0", transform: "translateY(20px)" }, "100%": { opacity: "1", transform: "translateY(0)" } },
        ticker:    { "0%": { transform: "translateX(100%)" }, "100%": { transform: "translateX(-100%)" } },
        scanner:   { "0%,100%": { transform: "translateY(0%)", opacity: "0" }, "50%": { transform: "translateY(100%)", opacity: "1" } },
        glowPulse: { "0%,100%": { boxShadow: "0 0 8px rgba(6,182,212,0.3)"  }, "50%":  { boxShadow: "0 0 24px rgba(6,182,212,0.8)" } },
        float:     { "0%,100%": { transform: "translateY(0px)" }, "50%": { transform: "translateY(-6px)" } },
        blink:     { "0%,100%": { opacity: "1" }, "50%": { opacity: "0" } },
        counterUp: { "0%": { opacity: "0", transform: "translateY(10px)" }, "100%": { opacity: "1", transform: "translateY(0)" } },
      },
      fontFamily: {
        mono: ["'JetBrains Mono'", "'Fira Code'", "monospace"],
        display: ["'Orbitron'", "sans-serif"],
      },
    },
  },
  plugins: [],
};
