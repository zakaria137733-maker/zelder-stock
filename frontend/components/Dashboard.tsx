"use client";
import { useQuery } from "@tanstack/react-query";
import { getSentiment, getAllSentiment, getSignals, getTransactions } from "@/lib/api";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { TrendingUp, TrendingDown, Minus, Activity, Users, DollarSign, BarChart2, type LucideIcon } from "lucide-react";
import AlertsPanel from "@/components/Alerts";
import Predictions from "@/components/Predictions";
import type { Signal, Trade } from "@/lib/types";

const card = { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: 16 };
const errorBanner = { background: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.35)", borderRadius: 8, padding: "10px 14px", fontSize: 12, color: "var(--red)", textAlign: "center" as const };
const dbBadge = (type: "influx" | "mongo") => ({
  fontSize: 9, padding: "2px 7px", borderRadius: 4, fontWeight: 600,
  background: type === "mongo" ? "var(--green-dim)" : "var(--accent-dim)",
  color: type === "mongo" ? "var(--green)" : "#60a5fa"
});

function KPI({ label, value, delta, up, icon: Icon }: {
  label: string;
  value: string | number;
  delta: string | number;
  up: boolean | null;
  icon: LucideIcon;
}) {
  const color = up === true ? "var(--green)" : up === false ? "var(--red)" : "var(--muted)";
  return (
    <div style={card}>
      <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 10 }}>
        <Icon size={11} />{label}
      </div>
      <div style={{ fontSize: 26, fontWeight: 600, color, marginBottom: 4, lineHeight: 1 }}>{value}</div>
      <div style={{ fontSize: 11, color, display: "flex", alignItems: "center", gap: 3 }}>
        {up === true ? <TrendingUp size={10} /> : up === false ? <TrendingDown size={10} /> : <Minus size={10} />}
        {delta}
      </div>
    </div>
  );
}

type TooltipProps = {
  active?: boolean;
  payload?: { value?: number }[];
  label?: string;
};

const CustomTooltip = ({ active, payload, label }: TooltipProps) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "var(--surface-2)", border: "1px solid var(--border-strong)", borderRadius: 8, padding: "8px 12px", fontSize: 11 }}>
      <div style={{ color: "var(--muted)", marginBottom: 4 }}>{new Date(label ?? 0).toLocaleTimeString()}</div>
      <div style={{ color: "var(--text)", fontWeight: 500 }}>Score: {payload[0]?.value?.toFixed(1)}</div>
    </div>
  );
};

export default function Dashboard({ ticker }: { ticker: string }) {
  const { data: sent, isError: sentError } = useQuery({ queryKey: ["sentiment", ticker], queryFn: () => getSentiment(ticker), refetchInterval: 60000 });
  const { data: allSent, isError: allSentError } = useQuery({ queryKey: ["allSentiment"], queryFn: getAllSentiment, refetchInterval: 60000 });
  const { data: signals, isError: signalsError } = useQuery({ queryKey: ["signals", ticker], queryFn: () => getSignals(ticker), refetchInterval: 30000 });
  const { data: txData, isError: txError } = useQuery({ queryKey: ["transactions", ticker], queryFn: () => getTransactions(ticker), refetchInterval: 30000 });

  const composite = sent?.composite ?? 50;
  const label = sent?.label ?? "neutral";
  const history = sent?.history ?? [];
  const trades = txData?.trades ?? [];
  const feed = signals ?? [];
  const scoreColor = composite >= 60 ? "var(--green)" : composite <= 40 ? "var(--red)" : "var(--amber)";
  const pos = sent?.breakdown?.positive ?? 0;
  const neg = sent?.breakdown?.negative ?? 0;
  const neu = sent?.breakdown?.neutral ?? 0;
  const total = pos + neg + neu || 1;

  return (
    <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 14 }}>

      {sentError || allSentError || signalsError || txError ? (
        <div style={errorBanner}>Backend unreachable — check the API</div>
      ) : null}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 }}>
        <KPI label="Sentiment Score" value={composite.toFixed(1)} delta={label} up={composite >= 60 ? true : composite <= 40 ? false : null} icon={Activity} />
        <KPI label="Signals Today" value={sent?.signal_count ?? 0} delta="from FinBERT" up={null} icon={BarChart2} />
        <KPI label="Tickers Tracked" value={allSent?.length ?? 7} delta="all markets" up={null} icon={Users} />
        <KPI label="Recent Trades" value={trades.length} delta={`${ticker} · 24h`} up={null} icon={DollarSign} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 12 }}>
        <div style={card}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
            <div style={{ fontSize: 13, fontWeight: 500 }}>Sentiment History — {ticker}</div>
            <span style={dbBadge("influx")}>InfluxDB</span>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={history}>
              <defs>
                <linearGradient id="sentGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={scoreColor} stopOpacity={0.3} />
                  <stop offset="100%" stopColor={scoreColor} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="time" tickFormatter={(v) => new Date(v).getHours() + ":00"} tick={{ fontSize: 10, fill: "var(--muted)" }} tickLine={false} axisLine={false} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "var(--muted)" }} tickLine={false} axisLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="value" stroke={scoreColor} strokeWidth={2} fill="url(#sentGrad)" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div style={card}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
            <div style={{ fontSize: 13, fontWeight: 500 }}>Composite</div>
            <span style={dbBadge("mongo")}>MongoDB</span>
          </div>
          <div style={{ textAlign: "center", padding: "16px 0 20px" }}>
            <div style={{ fontSize: 56, fontWeight: 700, color: scoreColor, lineHeight: 1, letterSpacing: "-2px" }}>{Math.round(composite)}</div>
            <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 6, textTransform: "capitalize" }}>{label} · {sent?.signal_count ?? 0} signals</div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {[
              { label: "Positive", val: Math.round(pos / total * 100), color: "#10b981" },
              { label: "Negative", val: Math.round(neg / total * 100), color: "#ef4444" },
              { label: "Neutral", val: Math.round(neu / total * 100), color: "#6b7280" },
            ].map((b) => (
              <div key={b.label} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11 }}>
                <span style={{ width: 44, color: "var(--muted)", fontSize: 10 }}>{b.label}</span>
                <div style={{ flex: 1, height: 5, background: "rgba(255,255,255,0.06)", borderRadius: 3, overflow: "hidden" }}>
                  <div style={{ height: "100%", width: `${b.val}%`, background: b.color, borderRadius: 3, transition: "width 0.6s" }} />
                </div>
                <span style={{ width: 28, textAlign: "right", color: "var(--muted)", fontSize: 10 }}>{b.val}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div style={card}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
            <div style={{ fontSize: 13, fontWeight: 500 }}>Live Signal Feed</div>
            <div style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 10, color: "#10b981" }}>
              <span style={{ width: 5, height: 5, borderRadius: "50%", background: "#10b981", display: "inline-block" }} />
              Real-time
            </div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 260, overflowY: "auto" }}>
            {signalsError ? (
              <div style={{ fontSize: 12, color: "var(--red)", textAlign: "center", padding: "20px 0" }}>Backend unreachable — check the API</div>
            ) : feed.length === 0 ? (
              <div style={{ fontSize: 12, color: "var(--muted)", textAlign: "center", padding: "20px 0" }}>No signals — run the collector</div>
            ) : null}
            {!signalsError && feed.slice(0, 8).map((s: Signal, i: number) => (
              <div key={i} style={{ padding: "9px 10px", borderRadius: 8, background: "var(--surface-2)", border: "1px solid var(--border)" }}>
                <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.4px", marginBottom: 3 }}>{s.source_name || s.source}</div>
                <div style={{ fontSize: 12, color: "var(--text)", lineHeight: 1.4, marginBottom: 6 }}>{s.title}</div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{
                    fontSize: 10, padding: "1px 7px", borderRadius: 20, fontWeight: 500,
                    background: s.label === "positive" ? "var(--green-dim)" : s.label === "negative" ? "var(--red-dim)" : "rgba(107,114,128,0.15)",
                    color: s.label === "positive" ? "var(--green)" : s.label === "negative" ? "var(--red)" : "var(--muted)"
                  }}>{s.label}</span>
                  <span style={{ fontSize: 10, color: "var(--muted)" }}>{s.age_hours?.toFixed(1)}h ago</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div style={card}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
            <div style={{ fontSize: 13, fontWeight: 500 }}>Recent Trades</div>
            <span style={dbBadge("influx")}>InfluxDB</span>
          </div>
          <div style={{ fontFamily: "monospace", fontSize: 10, color: "var(--muted)", background: "var(--surface-2)", borderRadius: 6, padding: "6px 10px", marginBottom: 10 }}>
            SELECT * FROM trades WHERE ticker=&apos;{ticker}&apos;
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4, maxHeight: 240, overflowY: "auto" }}>
            {trades.slice(0, 15).map((t: Trade, i: number) => (
              <div key={i} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "6px 8px", borderRadius: 6, background: "var(--surface-2)", fontSize: 11 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{
                    fontSize: 10, padding: "1px 6px", borderRadius: 4, fontWeight: 700,
                    background: t.side === "buy" ? "var(--green-dim)" : "var(--red-dim)",
                    color: t.side === "buy" ? "var(--green)" : "var(--red)"
                  }}>{t.side?.toUpperCase()}</span>
                  <span style={{ color: "var(--muted)" }}>{ticker}</span>
                </div>
                <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                  <span style={{ fontWeight: 600 }}>${t.price?.toFixed(2)}</span>
                  <span style={{ color: "var(--muted)", fontSize: 10 }}>{new Date(t.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <Predictions />
      <AlertsPanel />

    </div>
  );
}
