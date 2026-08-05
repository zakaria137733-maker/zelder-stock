"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import api from "@/lib/api";

type Mode = "login" | "register" | "admin";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>("login");
  const [form, setForm] = useState({ name: "", email: "", username: "", password: "", confirmPassword: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    setError("");
    if (mode === "admin") {
      if (!form.username || !form.password) return setError("Username and password required");
      setLoading(true);
      try {
        const { data } = await api.post("/api/admin/login", { username: form.username, password: form.password });
        localStorage.setItem("zs_token", data.token);
        localStorage.setItem("zs_user", JSON.stringify({ name: data.username, role: "admin" }));
        router.push("/");
      } catch (e: unknown) {
        const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
        setError(detail ?? "Something went wrong");
      } finally {
        setLoading(false);
      }
      return;
    }

    if (!form.email || !form.password) return setError("Email and password required");
    if (mode === "register" && form.password !== form.confirmPassword) return setError("Passwords don't match");

    setLoading(true);
    try {
      const endpoint = mode === "login" ? "/api/auth/login" : "/api/auth/register";
      const body = mode === "login"
        ? { email: form.email, password: form.password }
        : { name: form.name, email: form.email, password: form.password };

      const { data } = await api.post(endpoint, body);
      localStorage.setItem("zs_token", data.token);
      localStorage.setItem("zs_user", JSON.stringify({ name: data.name, email: data.email }));
      router.push("/");
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail ?? "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const inputStyle: React.CSSProperties = {
    width: "100%", padding: "10px 12px", borderRadius: 8,
    border: "1px solid rgba(255,255,255,0.1)",
    background: "rgba(255,255,255,0.05)",
    color: "#e8e6e1", fontSize: 13, outline: "none",
    marginBottom: 10,
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0d0f14", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div style={{ width: 380, background: "#13161e", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 16, padding: 32 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 28 }}>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: "linear-gradient(135deg, #3b82f6, #8b5cf6)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16 }}>⚡</div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: "#e8e6e1" }}>ZelderStock</div>
            <div style={{ fontSize: 10, color: "#6b7280" }}>Intelligence Platform</div>
          </div>
        </div>

        <div style={{ fontSize: 18, fontWeight: 600, color: "#e8e6e1", marginBottom: 6 }}>
          {mode === "login" ? "Welcome back" : mode === "register" ? "Create account" : "Admin sign in"}
        </div>
        <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 24 }}>
          {mode === "login" ? "Sign in to your ZelderStock account"
            : mode === "register" ? "Start monitoring market sentiment"
            : "Restricted to administrators"}
        </div>

        {mode === "admin" ? (
          <input style={inputStyle} placeholder="Admin username" value={form.username}
            onChange={e => setForm(f => ({ ...f, username: e.target.value }))} />
        ) : (
          mode === "register" && (
            <input style={inputStyle} placeholder="Full name" value={form.name}
              onChange={e => setForm(f => ({ ...f, name: e.target.value }))} />
          )
        )}
        {mode !== "admin" && (
          <input style={inputStyle} placeholder="Email address" type="email" value={form.email}
            onChange={e => setForm(f => ({ ...f, email: e.target.value }))} />
        )}
        <input style={inputStyle} placeholder="Password" type="password" value={form.password}
          onChange={e => setForm(f => ({ ...f, password: e.target.value }))} />
        {mode === "register" && (
          <input style={inputStyle} placeholder="Confirm password" type="password" value={form.confirmPassword}
            onChange={e => setForm(f => ({ ...f, confirmPassword: e.target.value }))} />
        )}

        {error && <div style={{ fontSize: 12, color: "#ef4444", marginBottom: 12 }}>{error}</div>}

        <button onClick={submit} disabled={loading} style={{
          width: "100%", padding: "11px", borderRadius: 8, border: "none",
          background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
          color: "#fff", fontSize: 13, fontWeight: 600, cursor: "pointer", marginBottom: 16
        }}>
          {loading ? "Please wait..." : mode === "login" ? "Sign In" : mode === "register" ? "Create Account" : "Admin Sign In"}
        </button>

        <div style={{ textAlign: "center", fontSize: 12, color: "#6b7280", marginBottom: 8 }}>
          {mode === "admin" ? (
            <>Looking for your account?{" "}
              <button onClick={() => setMode("login")}
                style={{ background: "none", border: "none", color: "#60a5fa", cursor: "pointer", fontSize: 12, fontWeight: 500 }}>
                Customer sign in
              </button></>
          ) : (
            <button onClick={() => setMode("admin")}
              style={{ background: "none", border: "none", color: "#60a5fa", cursor: "pointer", fontSize: 12, fontWeight: 500 }}>
              Admin sign in
            </button>
          )}
        </div>
        {mode !== "admin" && (
          <div style={{ textAlign: "center", fontSize: 12, color: "#6b7280" }}>
            {mode === "login" ? "Don't have an account? " : "Already have an account? "}
            <button onClick={() => setMode(mode === "login" ? "register" : "login")}
              style={{ background: "none", border: "none", color: "#60a5fa", cursor: "pointer", fontSize: 12, fontWeight: 500 }}>
              {mode === "login" ? "Register" : "Sign in"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}