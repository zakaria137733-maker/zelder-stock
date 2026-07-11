"use client";
import { useQuery } from "@tanstack/react-query";
import { getTransactions } from "@/lib/api";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

const card = { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: 16 };

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "var(--surface-2)", border: "1px solid var(--border-strong)", borderRadius: 8, padding: "8px 12px", fontSize: 11 }}>
      <div style={{ color: "var(--muted)", marginBottom: 4 }}>{label}</div>
      {payload.map((p: any) => (
        <div key={p.name} style={{ color: p.fill, marginBottom: 2 }}>{p.name}: {p.value}</div>
      ))}
    </div>
  );
};

export default function Transactions({ ticker }: { ticker: string }) {
  const { data: txData, isLoading } = useQuery({ queryKey: ["transactions", ticker], queryFn: () => getTransactions(ticker), refetchInterval: 30000 });
  const trades = txData?.trades ?? [];

  const hourly: Record<string, { buy: number; sell: number }> = {};
  trades.forEach((t: any) => {
    const h = new Date(t.time).getHours() + ":00";
    if (!hourly[h]) hourly[h] = { buy: 0, sell: 0 };
    if (t.side === "buy") hourly[h].buy++;
    else hourly[h].sell++;
  });
  const chartData = Object.entries(hourly).map(([time, v]) => ({ time, ...v }));

  const buyCount = trades.filter((t: any) => t.side === "buy").length;
  const sellCount = trades.filter((t: any) => t.side === "sell").length;
  const avgPrice = trades.length ? trades.reduce((s: number, t: any) => s + (t.price ?? 0), 0) / trades.length : 0;

  return (
    <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
        <div style={card}>
          <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 8 }}>Total Trades</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: "var(--text)" }}>{trades.length}</div>
          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>{ticker} · last 24h</div>
        </div>
        <div style={card}>
          <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 8 }}>Buy / Sell Split</div>
          <div style={{ fontSize: 28, fontWeight: 700 }}>
            <span style={{ color: "var(--green)" }}>{buyCount}</span>
            <span style={{ color: "var(--muted)", fontSize: 18, margin: "0 6px" }}>/</span>
            <span style={{ color: "var(--red)" }}>{sellCount}</span>
          </div>
          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>buys vs sells</div>
        </div>
        <div style={card}>
          <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 8 }}>Avg Price</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: "var(--text)" }}>${avgPrice.toFixed(2)}</div>
          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>{ticker} average</div>
        </div>
      </div>

      <div style={card}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
          <div style={{ fontSize: 13, fontWeight: 500 }}>Buy / Sell Volume by Hour</div>
          <span style={{ fontSize: 9, padding: "2px 7px", borderRadius: 4, fontWeight: 600, background: "var(--accent-dim)", color: "#60a5fa" }}>InfluxDB</span>
        </div>
        <div style={{ fontFamily: "monospace", fontSize: 10, color: "var(--muted)", background: "var(--surface-2)", borderRadius: 6, padding: "6px 10px", marginBottom: 14 }}>
          from(bucket:&quot;stock_trades&quot;) |&gt; range(start: -24h) |&gt; filter(fn: (r) =&gt; r.ticker == &quot;{ticker}&quot;)
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="time" tick={{ fontSize: 10, fill: "var(--muted)" }} tickLine={false} axisLine={false} />
            <YAxis tick={{ fontSize: 10, fill: "var(--muted)" }} tickLine={false} axisLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="buy" fill="#10b981" opacity={0.8} radius={[3, 3, 0, 0]} name="Buy" />
            <Bar dataKey="sell" fill="#ef4444" opacity={0.8} radius={[3, 3, 0, 0]} name="Sell" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={card}>
        <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 14 }}>Trade Ledger</div>
        {isLoading ? (
          <div style={{ fontSize: 12, color: "var(--muted)", textAlign: "center", padding: 30 }}>Loading trades...</div>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border)" }}>
                {["Time", "Ticker", "Side", "Price", "Date"].map(h => (
                  <th key={h} style={{ textAlign: "left", fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.5px", fontWeight: 500, padding: "0 8px 10px" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {trades.map((t: any, i: number) => (
                <tr key={i} style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "8px", fontFamily: "monospace", fontSize: 11, color: "var(--muted)" }}>{new Date(t.time).toLocaleTimeString()}</td>
                  <td style={{ padding: "8px", fontWeight: 600 }}>{t.ticker}</td>
                  <td style={{ padding: "8px" }}>
                    <span style={{ fontSize: 10, padding: "2px 7px", borderRadius: 4, fontWeight: 700, background: t.side === "buy" ? "var(--green-dim)" : "var(--red-dim)", color: t.side === "buy" ? "var(--green)" : "var(--red)" }}>
                      {t.side?.toUpperCase()}
                    </span>
                  </td>
                  <td style={{ padding: "8px", fontWeight: 600 }}>${t.price?.toFixed(2)}</td>
                  <td style={{ padding: "8px", color: "var(--muted)" }}>{new Date(t.time).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
