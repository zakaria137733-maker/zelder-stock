"use client";

const TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"];

export default function Topbar({ title, ticker, onTicker }: { title: string; ticker: string; onTicker: (t: string) => void }) {
  return (
    <header style={{ background: "var(--surface)", borderBottom: "1px solid var(--border)", padding: "0 20px", height: 52, display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
      <div>
        <div style={{ fontSize: 14, fontWeight: 500 }}>{title}</div>
        <div style={{ fontSize: 11, color: "var(--muted)", display: "flex", alignItems: "center", gap: 6, marginTop: 1 }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#10b981", display: "inline-block", animation: "pulse 2s infinite" }} />
          Live · refreshes every 30s
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        {TICKERS.map((t) => (
          <button key={t} onClick={() => onTicker(t)} style={{
            fontSize: 11, padding: "4px 10px", borderRadius: 6,
            border: ticker === t ? "1px solid rgba(59,130,246,0.5)" : "1px solid var(--border-strong)",
            background: ticker === t ? "var(--accent-dim)" : "transparent",
            color: ticker === t ? "#60a5fa" : "var(--muted)",
            cursor: "pointer", fontWeight: ticker === t ? 600 : 400, transition: "all 0.1s"
          }}>{t}</button>
        ))}
      </div>
    </header>
  );
}
