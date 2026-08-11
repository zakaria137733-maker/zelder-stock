"use client";
import { LayoutDashboard, TrendingUp, Users, ArrowLeftRight, Rss, Settings, Zap } from "lucide-react";
import { useRouter } from "next/navigation";

const nav = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "sentiment", label: "Sentiment Feed", icon: Rss },
  { id: "market", label: "Market Data", icon: TrendingUp },
  { id: "customers", label: "Customers", icon: Users, db: "Mongo" },
  { id: "transactions", label: "Transactions", icon: ArrowLeftRight, db: "Influx" },
  { id: "settings", label: "Settings", icon: Settings },
];

export default function Sidebar({ active, onNav }: { active: string; onNav: (v: string) => void }) {
  const router = useRouter();

  const logout = () => {
    localStorage.removeItem("zs_token");
    localStorage.removeItem("zs_user");
    router.push("/login");
  };

  const getUser = () => {
    try {
      return JSON.parse(localStorage.getItem("zs_user") || "{}");
    } catch { return {}; }
  };

  return (
    <aside style={{ width: 220, flexShrink: 0, background: "var(--surface)", borderRight: "1px solid var(--border)", display: "flex", flexDirection: "column", height: "100vh", position: "sticky", top: 0 }}>
      <div style={{ padding: "18px 16px 14px", borderBottom: "1px solid var(--border)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, fontWeight: 600, fontSize: 15, letterSpacing: "-0.3px" }}>
          <div style={{ width: 26, height: 26, borderRadius: 6, background: "linear-gradient(135deg, #3b82f6, #8b5cf6)", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Zap size={14} color="#fff" />
          </div>
          ZelderStock
        </div>
        <div style={{ fontSize: 10, color: "var(--muted)", letterSpacing: "0.6px", textTransform: "uppercase", marginTop: 4, paddingLeft: 34 }}>Intelligence Platform</div>
      </div>

      <nav style={{ flex: 1, padding: "10px 0" }}>
        <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.6px", padding: "8px 16px 4px" }}>Analytics</div>
        {nav.slice(0, 3).map((item) => (
          <button key={item.id} onClick={() => onNav(item.id)} style={{
            width: "100%", display: "flex", alignItems: "center", gap: 8, padding: "7px 16px",
            fontSize: 13, cursor: "pointer", border: "none", borderLeft: "2px solid",
            borderLeftColor: active === item.id ? "#3b82f6" : "transparent",
            background: active === item.id ? "var(--accent-dim)" : "transparent",
            color: active === item.id ? "#60a5fa" : "var(--muted)",
            fontWeight: active === item.id ? 500 : 400, transition: "all 0.12s", textAlign: "left"
          }}>
            <item.icon size={14} />{item.label}
          </button>
        ))}

        <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.6px", padding: "12px 16px 4px" }}>Data Sources</div>
        {nav.slice(3, 5).map((item) => (
          <button key={item.id} onClick={() => onNav(item.id)} style={{
            width: "100%", display: "flex", alignItems: "center", gap: 8, padding: "7px 16px",
            fontSize: 13, cursor: "pointer", border: "none", borderLeft: "2px solid",
            borderLeftColor: active === item.id ? "#3b82f6" : "transparent",
            background: active === item.id ? "var(--accent-dim)" : "transparent",
            color: active === item.id ? "#60a5fa" : "var(--muted)",
            fontWeight: active === item.id ? 500 : 400, transition: "all 0.12s", textAlign: "left"
          }}>
            <item.icon size={14} />{item.label}
            {item.db && (
              <span style={{
                marginLeft: "auto", fontSize: 9, padding: "2px 6px", borderRadius: 4, fontWeight: 600,
                background: item.db === "Mongo" ? "rgba(16,185,129,0.15)" : "rgba(59,130,246,0.15)",
                color: item.db === "Mongo" ? "#10b981" : "#60a5fa"
              }}>{item.db}</span>
            )}
          </button>
        ))}

        <div style={{ fontSize: 10, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.6px", padding: "12px 16px 4px" }}>System</div>
        {nav.slice(5).map((item) => (
          <button key={item.id} onClick={() => onNav(item.id)} style={{
            width: "100%", display: "flex", alignItems: "center", gap: 8, padding: "7px 16px",
            fontSize: 13, cursor: "pointer", border: "none", borderLeft: "2px solid",
            borderLeftColor: active === item.id ? "#3b82f6" : "transparent",
            background: active === item.id ? "var(--accent-dim)" : "transparent",
            color: active === item.id ? "#60a5fa" : "var(--muted)",
            fontWeight: active === item.id ? 500 : 400, transition: "all 0.12s", textAlign: "left"
          }}>
            <item.icon size={14} />{item.label}
          </button>
        ))}
      </nav>

      <div style={{ padding: "12px 16px", borderTop: "1px solid var(--border)" }}>
        <div style={{ fontSize: 10, color: "var(--muted)" }}>v1.0.0 · All systems operational</div>
        <div style={{ display: "flex", alignItems: "center", gap: 5, marginTop: 4, marginBottom: 8 }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#10b981", display: "inline-block" }} />
          <span style={{ fontSize: 10, color: "#10b981" }}>Live data active</span>
        </div>
        {getUser().name && (
          <div style={{ fontSize: 11, color: "var(--text)", fontWeight: 500, marginBottom: 8 }}>
            {getUser().name}
          </div>
        )}
        <button onClick={logout} style={{
          width: "100%", padding: "7px", borderRadius: 7,
          border: "1px solid var(--border-strong)",
          background: "transparent", color: "var(--muted)",
          fontSize: 11, cursor: "pointer"
        }}
          onMouseEnter={e => { e.currentTarget.style.background = "var(--red-dim)"; e.currentTarget.style.color = "var(--red)"; }}
          onMouseLeave={e => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "var(--muted)"; }}
        >
          Sign Out
        </button>
      </div>
    </aside>
  );
}