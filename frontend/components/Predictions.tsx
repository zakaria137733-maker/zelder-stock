"use client";
import { useQuery } from "@tanstack/react-query";
import api from "@/lib/api";
import type { Prediction } from "@/lib/types";

const getPredictions = () => api.get("/api/predictions/").then(r => r.data);

const signalColor = (signal: string) =>
  signal === "BUY" ? "var(--green)" : signal === "SELL" ? "var(--red)" : signal === "NO_SIGNAL" ? "var(--muted)" : "var(--amber)";

const signalBg = (signal: string) =>
  signal === "BUY" ? "var(--green-dim)" : signal === "SELL" ? "var(--red-dim)" : signal === "NO_SIGNAL" ? "var(--surface-2)" : "var(--amber-dim)";

export default function Predictions() {
  const { data: predictions, isLoading } = useQuery({
    queryKey: ["predictions"],
    queryFn: getPredictions,
    refetchInterval: 300000,
  });

  const list = predictions ?? [];

  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: 16 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
        <div style={{ fontSize: 13, fontWeight: 500 }}>LSTM Price Predictions</div>
        <span style={{ fontSize: 9, padding: "2px 7px", borderRadius: 4, fontWeight: 600, background: "var(--accent-dim)", color: "#60a5fa" }}>
          LSTM Ensemble · 7 tickers
        </span>
      </div>

      <div style={{ fontSize: 10, color: "var(--muted)", marginBottom: 12, padding: "6px 10px", background: "var(--surface-2)", borderRadius: 6 }}>
        Predicts 5-day forward price direction from a 10-day sentiment + price window
      </div>

      <div style={{ fontSize: 9, color: "var(--muted)", marginBottom: 12 }}>
        NO_SIGNAL means walk-forward tests found no edge over momentum (McNemar p ≥ 0.05); prob_up is still shown for transparency.
      </div>

      {isLoading ? (
        <div style={{ fontSize: 12, color: "var(--muted)", textAlign: "center", padding: 20 }}>Loading predictions...</div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(7,1fr)", gap: 8 }}>
          {list.map((p: Prediction) => (
            <div key={p.ticker} style={{ padding: "10px 8px", borderRadius: 8, background: "var(--surface-2)", border: "1px solid var(--border)", textAlign: "center" }}>
              <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 6 }}>{p.ticker}</div>
              <div style={{
                fontSize: 11, fontWeight: 700, padding: "3px 8px", borderRadius: 6, marginBottom: 6,
                background: signalBg(p.signal), color: signalColor(p.signal)
              }}>{p.signal}</div>
              <div style={{ fontSize: 10, color: "var(--muted)" }}>{p.confidence_pct}</div>
              <div style={{ marginTop: 6, height: 3, borderRadius: 2, background: "rgba(255,255,255,0.06)", overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${p.prob_up * 100}%`, background: "var(--green)", borderRadius: 2 }} />
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "var(--muted)", marginTop: 3 }}>
                <span>↑{(p.prob_up * 100).toFixed(0)}%</span>
                <span>↓{(p.prob_down * 100).toFixed(0)}%</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}