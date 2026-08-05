"use client";
import { useQuery } from "@tanstack/react-query";
import { getAllSentiment, getSentiment } from "@/lib/api";
import api from "@/lib/api";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, ComposedChart, Area } from "recharts";
import type { SentimentHistoryPoint, PriceHistoryPoint } from "@/lib/types";

const getAllPrices = () => api.get("/api/prices/").then(r => r.data);
const getPriceHistory = (ticker: string) => api.get(`/api/prices/${ticker}/history`, { params: { hours: 24 } }).then(r => r.data);

const COLORS: Record<string, string> = {
  AAPL: "#3b82f6", TSLA: "#ef4444", NVDA: "#10b981",
  MSFT: "#f59e0b", GOOGL: "#8b5cf6", AMZN: "#06b6d4", META: "#ec4899",
};
const ALL_TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"];
const card = { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: 16 };

type TooltipProps = {
  active?: boolean;
  payload?: { dataKey?: string; stroke?: string; fill?: string; value?: number | string }[];
  label?: string | number;
};

const CustomTooltip = ({ active, payload, label }: TooltipProps) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "var(--surface-2)", border: "1px solid var(--border-strong)", borderRadius: 8, padding: "8px 12px", fontSize: 11 }}>
      <div style={{ color: "var(--muted)", marginBottom: 6 }}>{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.stroke || p.fill, marginBottom: 2 }}>
          {p.dataKey}: {typeof p.value === "number" ? p.value.toFixed(2) : p.value}
        </div>
      ))}
    </div>
  );
};

function CorrelationChart({ ticker }: { ticker: string }) {
  const { data: sentData } = useQuery({ queryKey: ["sentiment", ticker], queryFn: () => getSentiment(ticker) });
  const { data: priceData } = useQuery({ queryKey: ["priceHistory", ticker], queryFn: () => getPriceHistory(ticker) });

  const sentMap: Record<string, number> = {};
  (sentData?.history ?? []).forEach((p: SentimentHistoryPoint) => {
    const h = new Date(p.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    sentMap[h] = p.value;
  });

  const merged = (priceData ?? []).map((p: PriceHistoryPoint) => {
    const h = new Date(p.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return { time: h, price: p.price, sentiment: sentMap[h] ?? null };
  }).filter((p: { time: string; price: number; sentiment: number | null }) => p.price);

  return (
    <div style={card}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
        <div style={{ fontSize: 13, fontWeight: 500 }}>Price vs Sentiment — {ticker}</div>
        <span style={{ fontSize: 9, padding: "2px 7px", borderRadius: 4, fontWeight: 600, background: "var(--accent-dim)", color: "#60a5fa" }}>InfluxDB · Yahoo Finance</span>
      </div>
      {merged.length === 0 ? (
        <div style={{ fontSize: 12, color: "var(--muted)", textAlign: "center", padding: 40 }}>Run free_collect.py to load price history</div>
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={merged}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="time" tick={{ fontSize: 10, fill: "var(--muted)" }} tickLine={false} axisLine={false} interval={4} />
            <YAxis yAxisId="price" tick={{ fontSize: 10, fill: "var(--muted)" }} tickLine={false} axisLine={false} tickFormatter={v => `$${v.toFixed(0)}`} />
            <YAxis yAxisId="sent" orientation="right" domain={[0, 100]} tick={{ fontSize: 10, fill: "var(--muted)" }} tickLine={false} axisLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Legend iconType="circle" iconSize={7} wrapperStyle={{ fontSize: 11, color: "var(--muted)" }} />
            <Line yAxisId="price" type="monotone" dataKey="price" stroke={COLORS[ticker]} strokeWidth={2} dot={false} name="Price ($)" />
            <Area yAxisId="sent" type="monotone" dataKey="sentiment" stroke="#6b7280" strokeWidth={1} fill="rgba(107,114,128,0.08)" dot={false} name="Sentiment" strokeDasharray="4 2" />
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

export default function MarketData() {
  const { data: allSent } = useQuery({ queryKey: ["allSentiment"], queryFn: getAllSentiment, refetchInterval: 60000 });
  const { data: allPrices } = useQuery({ queryKey: ["allPrices"], queryFn: getAllPrices, refetchInterval: 300000 });

  const sentMap = Object.fromEntries((allSent ?? []).map((s: { ticker: string; composite?: number; label?: string }) => [s.ticker, s]));
  const priceMap = Object.fromEntries((allPrices ?? []).map((p: { ticker: string; price?: number; change_pct?: number }) => [p.ticker, p]));

  const histories = ALL_TICKERS.map(t => ({
    ticker: t,
    // eslint-disable-next-line react-hooks/rules-of-hooks
    data: useQuery({ queryKey: ["sentiment", t], queryFn: () => getSentiment(t) }).data?.history ?? []
  }));

  const histMap: Record<string, Record<string, string | number>> = {};
  histories.forEach(({ ticker, data }) => {
    data.forEach((p: SentimentHistoryPoint) => {
      const t = new Date(p.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      if (!histMap[t]) histMap[t] = { time: t };
      histMap[t][ticker] = p.value;
    });
  });
  const chartData = Object.values(histMap).slice(-24);

  return (
    <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 14 }}>
      {/* Ticker cards with real prices */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(7,1fr)", gap: 8 }}>
        {ALL_TICKERS.map((t) => {
          const s = sentMap[t];
          const p = priceMap[t];
          const score = s?.composite ?? 50;
          const color = score >= 60 ? "var(--green)" : score <= 40 ? "var(--red)" : "var(--amber)";
          const priceUp = (p?.change_pct ?? 0) >= 0;
          return (
            <div key={t} style={{ ...card, padding: 12 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                <span style={{ fontWeight: 700, fontSize: 12 }}>{t}</span>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: COLORS[t], display: "inline-block" }} />
              </div>
              {p?.price ? (
                <>
                  <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 2 }}>${p.price}</div>
                  <div style={{ fontSize: 10, color: priceUp ? "var(--green)" : "var(--red)", marginBottom: 8, fontWeight: 600 }}>
                    {priceUp ? "▲" : "▼"} {Math.abs(p.change_pct).toFixed(2)}%
                  </div>
                </>
              ) : (
                <div style={{ fontSize: 10, color: "var(--muted)", marginBottom: 8 }}>Price loading...</div>
              )}
              <div style={{ fontSize: 10, color: "var(--muted)", marginBottom: 3 }}>Sentiment</div>
              <div style={{ fontSize: 22, fontWeight: 700, color, lineHeight: 1 }}>{score.toFixed(0)}</div>
              <div style={{ fontSize: 9, color: "var(--muted)", marginTop: 3, textTransform: "capitalize" }}>{s?.label ?? "neutral"}</div>
            </div>
          );
        })}
      </div>

      {/* Price vs Sentiment correlation for selected ticker */}
      <CorrelationChart ticker="AAPL" />
      <CorrelationChart ticker="NVDA" />

      {/* Multi-ticker sentiment chart */}
      <div style={card}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
          <div style={{ fontSize: 13, fontWeight: 500 }}>24h Sentiment Correlation — All Tickers</div>
          <span style={{ fontSize: 9, padding: "2px 7px", borderRadius: 4, fontWeight: 600, background: "var(--accent-dim)", color: "#60a5fa" }}>InfluxDB</span>
        </div>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="time" tick={{ fontSize: 10, fill: "var(--muted)" }} tickLine={false} axisLine={false} interval={3} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "var(--muted)" }} tickLine={false} axisLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Legend iconType="circle" iconSize={7} wrapperStyle={{ fontSize: 11, color: "var(--muted)" }} />
            {ALL_TICKERS.map((t) => (
              <Line key={t} type="monotone" dataKey={t} stroke={COLORS[t]} strokeWidth={1.5} dot={false} connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
