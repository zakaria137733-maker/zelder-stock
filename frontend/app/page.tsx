"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Sidebar from "@/components/Sidebar";
import Topbar from "@/components/Topbar";
import Dashboard from "@/components/Dashboard";
import SentimentFeed from "@/components/SentimentFeed";
import MarketData from "@/components/MarketData";
import Customers from "@/components/Customers";
import Transactions from "@/components/Transactions";

const TITLES: Record<string, string> = {
  dashboard: "Market Overview",
  sentiment: "Sentiment Feed",
  market: "Market Data",
  customers: "Customer Database · MongoDB",
  transactions: "Stock Transactions · InfluxDB",
  settings: "Settings",
};

export default function Home() {
  const [view, setView] = useState("dashboard");
  const [ticker, setTicker] = useState("AAPL");
  const router = useRouter();

  useEffect(() => {
    const token = localStorage.getItem("zs_token");
    if (!token) router.push("/login");
  }, []);

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
      <Sidebar active={view} onNav={setView} />
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <Topbar title={TITLES[view] ?? view} ticker={ticker} onTicker={setTicker} />
        <main style={{ flex: 1, overflowY: "auto" }}>
          {view === "dashboard" && <Dashboard ticker={ticker} />}
          {view === "sentiment" && <SentimentFeed ticker={ticker} />}
          {view === "market" && <MarketData />}
          {view === "customers" && <Customers />}
          {view === "transactions" && <Transactions ticker={ticker} />}
          {view === "settings" && (
            <div style={{ padding: 20 }}>
              <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 12, padding: 40, textAlign: "center", color: "var(--muted)", fontSize: 13 }}>
                Settings coming soon
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}