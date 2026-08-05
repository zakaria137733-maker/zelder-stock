"use client";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getCustomers } from "@/lib/api";
import api from "@/lib/api";
import { useState } from "react";
import { Plus, X } from "lucide-react";

const card = { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: 20 };
const riskColor = (r: string) => r === "conservative"
  ? { background: "var(--green-dim)", color: "var(--green)" }
  : r === "aggressive"
  ? { background: "var(--red-dim)", color: "var(--red)" }
  : { background: "var(--amber-dim)", color: "var(--amber)" };

const scoreColor = (s: number) => s >= 70 ? "var(--green)" : s >= 50 ? "var(--amber)" : "var(--red)";

type Customer = {
  id: string;
  name: string;
  email: string;
  portfolio_value: number;
  sentiment_score: number;
  risk_profile: string;
  watchlist: string[];
  created_at: string;
};

type CustomerInput = {
  name: string;
  email: string;
  portfolio_value: number;
  risk_profile: string;
  watchlist: string[];
};

function Avatar({ name }: { name: string }) {
  const initials = name.split(" ").map((n: string) => n[0]).join("").slice(0, 2);
  return (
    <div style={{ width: 28, height: 28, borderRadius: "50%", background: "var(--accent-dim)", color: "#60a5fa", fontSize: 10, fontWeight: 600, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
      {initials}
    </div>
  );
}

function AddCustomerModal({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [form, setForm] = useState({ name: "", email: "", portfolio_value: "", risk_profile: "moderate", watchlist: "AAPL,NVDA" });
  const mutation = useMutation({
    mutationFn: (data: CustomerInput) => api.post("/api/customers/", data).then(r => r.data),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["customers"] }); onClose(); }
  });

  const submit = () => {
    if (!form.name || !form.email) return;
    mutation.mutate({
      name: form.name, email: form.email,
      portfolio_value: parseFloat(form.portfolio_value) || 0,
      risk_profile: form.risk_profile,
      watchlist: form.watchlist.split(",").map(t => t.trim().toUpperCase()).filter(Boolean)
    });
  };

  const inputStyle = { width: "100%", padding: "8px 10px", borderRadius: 7, border: "1px solid var(--border-strong)", background: "var(--surface-2)", color: "var(--text)", fontSize: 12, outline: "none" };
  const labelStyle = { fontSize: 11, color: "var(--muted)", marginBottom: 5, display: "block" };

  return (
    <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.7)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 50 }}>
      <div style={{ background: "var(--surface)", border: "1px solid var(--border-strong)", borderRadius: 14, padding: 24, width: 400 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
          <div style={{ fontSize: 14, fontWeight: 600 }}>Add Customer</div>
          <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", color: "var(--muted)" }}><X size={16} /></button>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div><label style={labelStyle}>Full Name *</label><input style={inputStyle} value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))} placeholder="Sarah Chen" /></div>
          <div><label style={labelStyle}>Email *</label><input style={inputStyle} value={form.email} onChange={e => setForm(f => ({ ...f, email: e.target.value }))} placeholder="sarah@example.com" /></div>
          <div><label style={labelStyle}>Portfolio Value (USD)</label><input style={inputStyle} type="number" value={form.portfolio_value} onChange={e => setForm(f => ({ ...f, portfolio_value: e.target.value }))} placeholder="50000" /></div>
          <div>
            <label style={labelStyle}>Risk Profile</label>
            <select style={inputStyle} value={form.risk_profile} onChange={e => setForm(f => ({ ...f, risk_profile: e.target.value }))}>
              <option value="conservative">Conservative</option>
              <option value="moderate">Moderate</option>
              <option value="aggressive">Aggressive</option>
            </select>
          </div>
          <div><label style={labelStyle}>Watchlist (comma separated)</label><input style={inputStyle} value={form.watchlist} onChange={e => setForm(f => ({ ...f, watchlist: e.target.value }))} placeholder="AAPL, TSLA, NVDA" /></div>
        </div>
        <div style={{ display: "flex", gap: 8, marginTop: 20 }}>
          <button onClick={onClose} style={{ flex: 1, padding: "9px", borderRadius: 8, border: "1px solid var(--border-strong)", background: "transparent", color: "var(--muted)", fontSize: 12, cursor: "pointer" }}>Cancel</button>
          <button onClick={submit} disabled={mutation.isPending} style={{ flex: 2, padding: "9px", borderRadius: 8, border: "none", background: "#3b82f6", color: "#fff", fontSize: 12, fontWeight: 500, cursor: "pointer" }}>
            {mutation.isPending ? "Adding..." : "Add Customer"}
          </button>
        </div>
        {mutation.isError && <div style={{ fontSize: 11, color: "var(--red)", marginTop: 8 }}>Error — email may already exist</div>}
      </div>
    </div>
  );
}

export default function Customers() {
  const [showAdd, setShowAdd] = useState(false);
  const { data: customers, isLoading, error } = useQuery({ queryKey: ["customers"], queryFn: getCustomers });
  const status = (error as { response?: { status?: number } })?.response?.status;
  const isAdmin = !error;

  if (error) {
    const adminRequired = status === 401 || status === 403;
    return (
      <div style={{ padding: 20 }}>
        <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: 40, textAlign: "center" }}>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>
            {adminRequired ? "Admin access required" : "Unable to load customers"}
          </div>
          <div style={{ fontSize: 12, color: "var(--muted)", maxWidth: 420, margin: "0 auto" }}>
            {adminRequired
              ? "The customer database contains personal information (names, emails, portfolio values) and is only visible to administrators. Sign in with an admin account to view it."
              : "The customer database could not be loaded. Please try again later."}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: 20 }}>
      {showAdd && <AddCustomerModal onClose={() => setShowAdd(false)} />}
      <div style={card}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 500 }}>Customer Database</div>
            <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>{customers?.length ?? 0} customers · sorted by sentiment score</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 9, padding: "2px 7px", borderRadius: 4, fontWeight: 600, background: "var(--green-dim)", color: "var(--green)" }}>MongoDB</span>
            {isAdmin && (
              <button onClick={() => setShowAdd(true)} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12, padding: "6px 12px", borderRadius: 8, border: "none", background: "#3b82f6", color: "#fff", cursor: "pointer", fontWeight: 500 }}>
                <Plus size={13} />Add Customer
              </button>
            )}
          </div>
        </div>
        <div style={{ fontFamily: "monospace", fontSize: 10, color: "var(--muted)", background: "var(--surface-2)", borderRadius: 6, padding: "6px 10px", marginBottom: 14 }}>
          db.customers.find({"{"}{"}"}).sort({"{"}sentiment_score: -1{"}"})
        </div>
        {isLoading ? (
          <div style={{ fontSize: 12, color: "var(--muted)", textAlign: "center", padding: 40 }}>Loading customers...</div>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border)" }}>
                {["Name", "Email", "Portfolio", "Sentiment", "Risk", "Watchlist", "Joined"].map(h => (
                  <th key={h} style={{ textAlign: "left", fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.5px", fontWeight: 500, padding: "0 8px 10px" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(customers ?? []).map((c: Customer) => (
                <tr key={c.id} style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "10px 8px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <Avatar name={c.name} />
                      <span style={{ fontWeight: 500, color: "var(--text)" }}>{c.name}</span>
                    </div>
                  </td>
                  <td style={{ padding: "10px 8px", color: "#60a5fa" }}>{c.email}</td>
                  <td style={{ padding: "10px 8px", fontWeight: 500 }}>${c.portfolio_value.toLocaleString()}</td>
                  <td style={{ padding: "10px 8px" }}>
                    <span style={{ fontWeight: 700, fontSize: 14, color: scoreColor(c.sentiment_score) }}>{c.sentiment_score}</span>
                  </td>
                  <td style={{ padding: "10px 8px" }}>
                    <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 20, fontWeight: 500, ...riskColor(c.risk_profile) }}>{c.risk_profile}</span>
                  </td>
                  <td style={{ padding: "10px 8px" }}>
                    <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                      {c.watchlist.map((t: string) => (
                        <span key={t} style={{ fontSize: 9, padding: "1px 6px", background: "rgba(255,255,255,0.06)", color: "var(--muted)", borderRadius: 4, fontWeight: 500 }}>{t}</span>
                      ))}
                    </div>
                  </td>
                  <td style={{ padding: "10px 8px", color: "var(--muted)" }}>{new Date(c.created_at).toLocaleDateString("en-US", { month: "short", year: "numeric" })}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
