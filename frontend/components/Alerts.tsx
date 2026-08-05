"use client";
import { useQuery } from "@tanstack/react-query";
import { getAlerts } from "@/lib/api";
import { AlertTriangle, TrendingDown, TrendingUp } from "lucide-react";
import type { Alert } from "@/lib/types";

export default function AlertsPanel() {
  const { data: alerts, refetch } = useQuery({
    queryKey: ["alerts"],
    queryFn: getAlerts,
    refetchInterval: 60000,
  });

  const list = alerts ?? [];

  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: 16 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 13, fontWeight: 500 }}>
          <AlertTriangle size={14} color="#f59e0b" />
          Sentiment Alerts
          {list.length > 0 && (
            <span style={{ fontSize: 10, padding: "1px 7px", borderRadius: 20, background: "var(--red-dim)", color: "var(--red)", fontWeight: 600 }}>
              {list.length} active
            </span>
          )}
        </div>
        <button onClick={() => refetch()} style={{ fontSize: 11, padding: "3px 10px", borderRadius: 6, border: "1px solid var(--border-strong)", background: "transparent", color: "var(--muted)", cursor: "pointer" }}>
          Refresh
        </button>
      </div>

      {list.length === 0 ? (
        <div style={{ fontSize: 12, color: "var(--muted)", textAlign: "center", padding: "20px 0" }}>
          No significant sentiment shifts detected
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {list.map((a: Alert, i: number) => (
            <div key={i} style={{
              padding: "10px 12px", borderRadius: 8,
              background: a.severity === "high" ? "var(--red-dim)" : "var(--amber-dim)",
              border: `1px solid ${a.severity === "high" ? "rgba(239,68,68,0.3)" : "rgba(245,158,11,0.3)"}`,
            }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 5 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  {a.direction === "down"
                    ? <TrendingDown size={13} color="var(--red)" />
                    : <TrendingUp size={13} color="var(--green)" />
                  }
                  <span style={{ fontWeight: 700, fontSize: 13 }}>{a.ticker}</span>
                  <span style={{
                    fontSize: 9, padding: "1px 6px", borderRadius: 4, fontWeight: 600, textTransform: "uppercase",
                    background: a.severity === "high" ? "var(--red-dim)" : "var(--amber-dim)",
                    color: a.severity === "high" ? "var(--red)" : "var(--amber)"
                  }}>{a.severity}</span>
                </div>
                <span style={{ fontSize: 12, fontWeight: 600, color: a.direction === "down" ? "var(--red)" : "var(--green)" }}>
                  {a.shift > 0 ? "+" : ""}{a.shift} pts
                </span>
              </div>
              <div style={{ fontSize: 12, color: "var(--text)", marginBottom: 4 }}>{a.message}</div>
              <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 10, color: "var(--muted)" }}>
                <span>{a.previous_score} → {a.current_score}</span>
                <span>·</span>
                <span>{new Date(a.triggered_at).toLocaleTimeString()}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}