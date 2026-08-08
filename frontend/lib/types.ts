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
  signal_gate?: boolean;
  evidence?: {
    n_windows?: number;
    lstm_acc?: number;
    momentum_acc?: number;
    majority_acc?: number;
    auc?: number | null;
    balanced_accuracy?: number;
    p_vs_momentum?: number;
    buy_threshold?: number;
    sell_threshold?: number;
  } | null;
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
