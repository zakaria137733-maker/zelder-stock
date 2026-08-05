"use client";
import { useQuery } from "@tanstack/react-query";
import { getSignals } from "@/lib/api";
import { useState } from "react";
import { ExternalLink } from "lucide-react";
import type { Signal } from "@/lib/types";

type Filter = "all" | "positive" | "negative" | "neutral";
const card = { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: 20 };

export default function SentimentFeed({ ticker }: { ticker: string }) {
  const [filter, setFilter] = useState<Filter>("all");
  const { data: signals, isLoading } = useQuery({ queryKey: ["signals", ticker], queryFn: () => getSignals(ticker), refetchInterval: 30000 });
  const filtered = (signals ?? []).filter((s: Signal) => filter === "all" || s.label === filter);

  return (
    <div style={{ padding: 20 }}>
      <div style={card}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 500 }}>Sentiment Analysis Feed — {ticker}</div>
          <div style={{ display: "flex", gap: 6 }}>
            {(["all", "positive", "negative", "neutral"] as Filter[]).map((f) => (
              <button key={f} onClick={() => setFilter(f)} style={{
                fontSize: 11, padding: "4px 12px", borderRadius: 6, cursor: "pointer",
                border: filter === f ? "1px solid rgba(59,130,246,0.4)" : "1px solid var(--border-strong)",
                background: filter === f ? "var(--accent-dim)" : "transparent",
                color: filter === f ? "#60a5fa" : "var(--muted)",
                textTransform: "capitalize", transition: "all 0.1s"
              }}>{f}</button>
            ))}
          </div>
        </div>

        {isLoading ? (
          <div style={{ fontSize: 12, color: "var(--muted)", textAlign: "center", padding: 40 }}>Fetching signals from NewsAPI...</div>
        ) : filtered.length === 0 ? (
          <div style={{ fontSize: 12, color: "var(--muted)", textAlign: "center", padding: 40 }}>No {filter === "all" ? "" : filter} signals found.</div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {filtered.map((s: Signal, i: number) => (
              <div key={i} style={{ padding: "12px 14px", borderRadius: 10, background: "var(--surface-2)", border: "1px solid var(--border)", display: "flex", gap: 12, alignItems: "flex-start" }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
                    <span style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.4px" }}>{s.source_name || s.source}</span>
                    <span style={{ fontSize: 9, padding: "1px 6px", background: "rgba(255,255,255,0.06)", color: "var(--muted)", borderRadius: 4, fontWeight: 500 }}>{s.ticker || ticker}</span>
                  </div>
                  <div style={{ fontSize: 13, color: "var(--text)", lineHeight: 1.5, marginBottom: 8 }}>{s.title}</div>
                  <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <span style={{
                      fontSize: 10, padding: "2px 8px", borderRadius: 20, fontWeight: 500,
                      background: s.label === "positive" ? "var(--green-dim)" : s.label === "negative" ? "var(--red-dim)" : "rgba(107,114,128,0.15)",
                      color: s.label === "positive" ? "var(--green)" : s.label === "negative" ? "var(--red)" : "var(--muted)"
                    }}>{s.label}</span>
                    <span style={{ fontSize: 10, color: "var(--muted)" }}>Score: {s.score?.toFixed(3)}</span>
                    <span style={{ fontSize: 10, color: "var(--muted)" }}>Confidence: {(s.confidence * 100).toFixed(0)}%</span>
                    <span style={{ fontSize: 10, color: "var(--muted)" }}>{s.age_hours?.toFixed(1)}h ago</span>
                  </div>
                </div>
                {s.url && (
                  <a href={s.url} target="_blank" rel="noopener noreferrer" style={{ color: "var(--muted)", flexShrink: 0, marginTop: 2, transition: "color 0.1s" }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = "#60a5fa")}
                    onMouseLeave={(e) => (e.currentTarget.style.color = "var(--muted)")}>
                    <ExternalLink size={14} />
                  </a>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
