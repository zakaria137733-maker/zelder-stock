"use client";
import { useQuery } from "@tanstack/react-query";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, XAxis, YAxis } from "recharts";
import { AlertTriangle, BarChart3, ShieldCheck } from "lucide-react";
import { getEvalReport } from "@/lib/api";
import type { EvalReportResponse } from "@/lib/types";

const card = { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: 20 };

const pct = (n?: number | null) => (typeof n === "number" ? `${(n * 100).toFixed(1)}%` : "—");

export default function Evaluation() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["evalReport"],
    queryFn: getEvalReport,
  });

  const status = (error as { response?: { status?: number } })?.response?.status;
  const adminRequired = status === 401 || status === 403;

  if (error) {
    return (
      <div style={{ padding: 20 }}>
        <div style={{ ...card, padding: 40, textAlign: "center" }}>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>
            {adminRequired ? "Admin access required" : "Unable to load evaluation report"}
          </div>
          <div style={{ fontSize: 12, color: "var(--muted)", maxWidth: 460, margin: "0 auto" }}>
            {adminRequired
              ? "The model evaluation report is visible to administrators only. Sign in with an admin account to view the honest numbers behind the predictions."
              : "The evaluation report could not be loaded. Please try again later."}
          </div>
        </div>
      </div>
    );
  }

  const report = data?.report as EvalReportResponse["report"] | null | undefined;
  const caveats: string[] = data?.caveats ?? [];

  return (
    <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ ...card, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <BarChart3 size={14} color="#60a5fa" />
          <div style={{ fontSize: 13, fontWeight: 500 }}>Honest Model Evaluation</div>
        </div>
        {data?.source ? (
          <span style={{ fontSize: 9, padding: "2px 7px", borderRadius: 4, fontWeight: 600, background: "var(--accent-dim)", color: "#60a5fa" }}>
            {data.source}
          </span>
        ) : null}
      </div>

      {isLoading ? (
        <div style={{ ...card, textAlign: "center", fontSize: 12, color: "var(--muted)", padding: 40 }}>
          Loading evaluation report...
        </div>
      ) : !report ? (
        <div style={{ ...card, textAlign: "center", padding: 40 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 6 }}>No evaluation report committed yet</div>
          <div style={{ fontSize: 12, color: "var(--muted)", maxWidth: 520, margin: "0 auto", lineHeight: 1.6 }}>
            Generate one and commit it so the served numbers stay visible:
            <code style={{ display: "block", marginTop: 8, fontFamily: "monospace", fontSize: 11, background: "var(--surface-2)", borderRadius: 6, padding: "8px 10px" }}>
              docker-compose exec api python scripts/eval_deployed.py AAPL --days 120 --json-out models/eval_report.json
            </code>
          </div>
        </div>
      ) : (
        <>
          <div style={{ ...card }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
              <div style={{ fontSize: 13, fontWeight: 500 }}>Accuracy vs baselines — {report.ticker ?? "model"}</div>
              <span style={{ fontSize: 9, padding: "2px 7px", borderRadius: 4, fontWeight: 600, background: "var(--green-dim)", color: "var(--green)" }}>
                deployed artifact
              </span>
            </div>
            {(() => {
              const chartData = [
                { name: "Model", value: report.accuracy ?? 0, fill: report.accuracy !== undefined && report.baseline !== undefined && report.accuracy >= report.baseline ? "var(--green)" : "var(--red)" },
                { name: "Majority baseline", value: report.baseline ?? 0, fill: "#60a5fa" },
                { name: "Coin flip", value: 0.5, fill: "#6b7280" },
              ];
              return (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={chartData} margin={{ top: 20, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                    <XAxis dataKey="name" tick={{ fontSize: 10, fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                    <YAxis domain={[0, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} tick={{ fontSize: 10, fill: "var(--muted)" }} tickLine={false} axisLine={false} />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                      {chartData.map((entry) => (
                        <Cell key={entry.name} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              );
            })()}
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--muted)", marginTop: 8 }}>
              <span>Read as: does the shipped model beat its own majority baseline on recent held-out data?</span>
              <span style={{ fontFamily: "monospace" }}>{report.windows ?? 0} windows · horizon {report.horizon ?? 5}d</span>
            </div>
          </div>

          <div style={card}>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 14 }}>Per-ticker report</div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  {["Ticker", "Windows", "Up / Down", "Accuracy", "Balanced", "AUC", "Baseline"].map((h) => (
                    <th key={h} style={{ textAlign: "left", fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.5px", fontWeight: 500, padding: "0 8px 10px" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "10px 8px", fontWeight: 600 }}>{report.ticker ?? "—"}</td>
                  <td style={{ padding: "10px 8px" }}>{report.windows ?? "—"}</td>
                  <td style={{ padding: "10px 8px", color: "var(--muted)" }}>
                    {typeof report.up_samples === "number" ? `${report.up_samples} / ${report.down_samples ?? "—"}` : "—"}
                  </td>
                  <td style={{ padding: "10px 8px" }}>
                    <span style={{
                      fontWeight: 700, fontSize: 14,
                      color: report.accuracy !== undefined && report.baseline !== undefined && report.accuracy >= report.baseline ? "var(--green)" : "var(--red)"
                    }}>{pct(report.accuracy)}</span>
                  </td>
                  <td style={{ padding: "10px 8px", color: "var(--muted)" }}>{pct(report.balanced_accuracy)}</td>
                  <td style={{ padding: "10px 8px", color: "var(--muted)" }}>{pct(report.auc)}</td>
                  <td style={{ padding: "10px 8px", color: "var(--muted)" }}>{pct(report.baseline)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </>
      )}

      <div style={{ ...card, borderLeft: "3px solid var(--amber)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
          <AlertTriangle size={14} color="var(--amber)" />
          <div style={{ fontSize: 13, fontWeight: 500 }}>Caveats — read before quoting any number</div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {caveats.map((c, i) => (
            <div key={i} style={{ display: "flex", gap: 8, fontSize: 12, color: "var(--muted)", lineHeight: 1.5 }}>
              <span style={{ color: "var(--amber)", flexShrink: 0 }}>•</span>
              <span>{c}</span>
            </div>
          ))}
          {caveats.length === 0 && (
            <div style={{ fontSize: 12, color: "var(--muted)" }}>No caveats recorded.</div>
          )}
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: "var(--muted)" }}>
        <ShieldCheck size={13} color="var(--green)" />
        Numbers come from the committed report artifact, not a freshly retrained model — the same serving path as /api/predictions.
      </div>
    </div>
  );
}
