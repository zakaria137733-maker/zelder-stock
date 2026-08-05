import axios from "axios";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
});

api.interceptors.request.use((config) => {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("zs_token");
    if (token) config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export const getSentiment = (ticker: string) =>
  api.get(`/api/sentiment/${ticker}`).then((r) => r.data);

export const getAllSentiment = () =>
  api.get("/api/sentiment/").then((r) => r.data);

export const getSignals = (ticker?: string) =>
  api.get("/api/signals", { params: ticker ? { ticker } : {} }).then((r) => r.data);

export const getCustomers = () =>
  api.get("/api/customers/").then((r) => r.data);

export const getTransactions = (ticker: string) =>
  api.get(`/api/transactions/${ticker}`).then((r) => r.data);

export default api;

export const getAlerts = () =>
  api.get("/api/alerts/").then((r) => r.data);
