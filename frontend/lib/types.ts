export type Signal = {
  source_name?: string;
  source?: string;
  ticker?: string;
  title: string;
  label: "positive" | "negative" | "neutral";
  score: number;
  confidence: number;
  age_hours: number;
  url?: string;
};

export type Trade = {
  time: string;
  ticker?: string;
  side: "buy" | "sell";
  price?: number;
};

export type Prediction = {
  ticker: string;
  signal: string;
  confidence_pct: string;
  prob_up: number;
  prob_down: number;
};

export type Alert = {
  ticker: string;
  severity: "high" | "medium";
  direction: "up" | "down";
  shift: number;
  message: string;
  previous_score: number;
  current_score: number;
  triggered_at: string;
};

export type SentimentHistoryPoint = { time: string; value: number };
export type PriceHistoryPoint = { time: string; price: number };
