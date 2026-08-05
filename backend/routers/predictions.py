import os
import json
import numpy as np
import torch
from fastapi import APIRouter
from services import influx
from services.lstm_predictor import SentimentLSTM, SCALER_PATH, SEQUENCE_LEN, FEATURES, compute_indicators

router = APIRouter(prefix="/api/predictions", tags=["predictions"])

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]
MODEL_DIR = "/app/models"
SEEDS = [42, 123, 7, 99, 2024]


def safe_compute_indicators(prices, volumes, dates):
    """Compute indicators safely — falls back to defaults if not enough data."""
    if len(prices) < 28:
        n = len(prices)
        return {
            "rsi": [50.0] * n,
            "macd": [0.0] * n,
            "bb_upper": list(prices),
            "bb_lower": list(prices),
            "bb_width": [0.0] * n,
            "ma20": list(prices),
            "ema20": list(prices),
            "ema50": list(prices),
            "atr": [0.0] * n,
            "vol_momentum": [1.0] * n,
            "adx": [25.0] * n,
            "obv": [0.0] * n,
            "stoch": [50.0] * n,
            "williams_r": [-50.0] * n,
            "cci": [0.0] * n,
            "vwap": list(prices),
            "day_of_week": [0.0] * n,
        }
    try:
        return compute_indicators(prices, volumes, dates)
    except Exception as e:
        print(f"Indicator error: {e}, falling back to defaults")
        n = len(prices)
        return {
            "rsi": [50.0] * n, "macd": [0.0] * n,
            "bb_upper": list(prices), "bb_lower": list(prices),
            "bb_width": [0.0] * n, "ma20": list(prices),
            "ema20": list(prices), "ema50": list(prices),
            "atr": [0.0] * n, "vol_momentum": [1.0] * n,
            "adx": [25.0] * n, "obv": [0.0] * n,
            "stoch": [50.0] * n, "williams_r": [-50.0] * n,
            "cci": [0.0] * n, "vwap": list(prices),
            "day_of_week": [0.0] * n,
        }


def ensemble_predict(ticker: str, recent_data: list) -> dict:
    if not os.path.exists(SCALER_PATH):
        return {"error": "Model not trained", "signal": "HOLD",
                "prob_up": 0.5, "prob_down": 0.5, "confidence": 0.5, "confidence_pct": "50%"}

    if len(recent_data) < SEQUENCE_LEN:
        return {"error": f"Need {SEQUENCE_LEN} data points", "signal": "HOLD",
                "prob_up": 0.5, "prob_down": 0.5, "confidence": 0.5, "confidence_pct": "50%"}

    prices = [r["price"] for r in recent_data]
    volumes = [r.get("volume", 0.0) for r in recent_data]
    dates = [r.get("time", "") for r in recent_data]

    indicators = safe_compute_indicators(prices, volumes, dates)

    window = []
    rows = recent_data[-SEQUENCE_LEN:]
    offset = len(recent_data) - SEQUENCE_LEN
    eps = 1e-8

    for i, row in enumerate(rows):
        idx = offset + i
        price = row["price"]
        prev_price = recent_data[offset + i - 1]["price"] if i > 0 else price
        price_change = (price - prev_price) / prev_price * 100 if prev_price else 0

        def ind(key, default):
            vals = indicators.get(key, [])
            return vals[idx] if idx < len(vals) else default

        features = [
            row.get("sentiment", 50.0) / 100.0,
            price_change,
            ind("rsi", 50.0) / 100.0,
            ind("macd", 0.0) / (price + eps) * 100,
            ind("bb_width", 0.0),
            ind("adx", 25.0) / 100.0,
            np.log1p(abs(ind("obv", 0.0))) * np.sign(ind("obv", 0.0)) / 25.0,
            ind("stoch", 50.0) / 100.0,
            ind("williams_r", -50.0) / 100.0,
            ind("cci", 0.0) / 200.0,
            row.get("spy_ret", 0.0),
            (row.get("vix", 20.0) - 20.0) / 10.0,
        ]
        window.append(np.clip(features, -5, 5))

    X = torch.tensor([window], dtype=torch.float32)
    probs = []

    for seed in SEEDS:
        model_path = f"{MODEL_DIR}/lstm_model_{seed}.pt"
        if not os.path.exists(model_path):
            continue
        try:
            model = SentimentLSTM()
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            with torch.no_grad():
                prob = float(model(X).view(-1).item())
                probs.append(prob)
        except Exception as e:
            print(f"Model {seed} error: {e}")

    if not probs:
        # fall back to single model
        single_path = f"{MODEL_DIR}/lstm_model.pt"
        if os.path.exists(single_path):
            try:
                model = SentimentLSTM()
                model.load_state_dict(torch.load(single_path, map_location="cpu"))
                model.eval()
                with torch.no_grad():
                    prob_up = float(model(X).view(-1).item())
                probs = [prob_up]
            except Exception as e:
                print(f"Single model error: {e}")

    if not probs:
        return {"ticker": ticker, "signal": "HOLD", "prob_up": 0.5,
                "prob_down": 0.5, "confidence": 0.5, "confidence_pct": "50%",
                "model_agreement": "N/A", "models_used": 0}

    prob_up = float(np.mean(probs))
    std = float(np.std(probs)) if len(probs) > 1 else 0.0
    signal = "BUY" if prob_up > 0.53 else "SELL" if prob_up < 0.47 else "HOLD"
    confidence = max(prob_up, 1 - prob_up)

    return {
        "ticker": ticker,
        "signal": signal,
        "prob_up": round(prob_up, 3),
        "prob_down": round(1 - prob_up, 3),
        "confidence": round(confidence, 3),
        "confidence_pct": f"{confidence*100:.0f}%",
        "model_agreement": f"{max(0, (1 - std/0.5)*100):.0f}%",
        "models_used": len(probs),
    }


def build_recent_data(price_history, sent_map, n=50):
    return [
        {
            "price": p["price"],
            "time": p["time"],
            "sentiment": sent_map.get(p["time"][:13], 50.0),
            "spy_ret": 0.0,
            "vix": 20.0,
        }
        for p in price_history[-n:]
    ]


@router.get("/{ticker}")
async def get_prediction(ticker: str):
    ticker = ticker.upper()
    price_history = influx.query_price_history(ticker, hours=720)
    sent_history = influx.query_sentiment_history(ticker, hours=720)
    sent_map = {s["time"][:13]: s["value"] for s in sent_history}
    recent_data = build_recent_data(price_history, sent_map)
    result = ensemble_predict(ticker, recent_data)
    result["ticker"] = ticker
    return result


@router.get("/")
async def get_all_predictions():
    results = []
    for ticker in TICKERS:
        price_history = influx.query_price_history(ticker, hours=720)
        sent_history = influx.query_sentiment_history(ticker, hours=720)
        sent_map = {s["time"][:13]: s["value"] for s in sent_history}
        recent_data = build_recent_data(price_history, sent_map)
        result = ensemble_predict(ticker, recent_data)
        result["ticker"] = ticker
        results.append(result)
    return results