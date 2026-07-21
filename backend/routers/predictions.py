from fastapi import APIRouter
from services import influx
from services.lstm_predictor import SentimentLSTM, SCALER_PATH, SEQUENCE_LEN, compute_indicators
import torch
import json
import os
import numpy as np

router = APIRouter(prefix="/api/predictions", tags=["predictions"])

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]
MODEL_DIR = "/app/models"
SEEDS = [42, 123, 99, 2024]  # exclude seed 7 which had 47% accuracy


def ensemble_predict(ticker: str, recent_data: list) -> dict:
    if not os.path.exists(SCALER_PATH):
        return {"error": "Model not trained", "signal": "unknown"}

    with open(SCALER_PATH) as f:
        scaler_params = json.load(f)

    if len(recent_data) < SEQUENCE_LEN:
        return {"error": f"Need {SEQUENCE_LEN} hours of data", "signal": "unknown"}

    prices = [r["price"] for r in recent_data]
    volumes = [r.get("volume", 0.0) for r in recent_data]
    dates = [r.get("time", "") for r in recent_data]

    if len(prices) >= 20:
        indicators = compute_indicators(prices, volumes, dates)
    else:
        indicators = {
            "rsi": [50.0] * len(prices),
            "macd": [0.0] * len(prices),
            "bb_upper": prices,
            "ma20": prices,
            "volatility": [0.0] * len(prices),
            "vol_momentum": [1.0] * len(prices),
            "day_of_week": [0.0] * len(prices),
        }

    window = []
    rows = recent_data[-SEQUENCE_LEN:]
    offset = len(recent_data) - SEQUENCE_LEN

    for i, row in enumerate(rows):
        idx = offset + i
        price = row["price"]
        prev_price = recent_data[offset + i - 1]["price"] if i > 0 else price
        price_change = (price - prev_price) / prev_price * 100 if prev_price else 0

        features = [
            price,
            price_change,
            row.get("volume", 0.0),
            indicators["rsi"][idx] if idx < len(indicators["rsi"]) else 50.0,
            indicators["macd"][idx] if idx < len(indicators["macd"]) else 0.0,
            indicators["bb_upper"][idx] if idx < len(indicators["bb_upper"]) else price,
            indicators["ma20"][idx] if idx < len(indicators["ma20"]) else price,
            indicators["volatility"][idx] if idx < len(indicators["volatility"]) else 0.0,
            indicators["vol_momentum"][idx] if idx < len(indicators["vol_momentum"]) else 1.0,
            0.0,
        ]

        normalized = []
        for f_idx, val in enumerate(features):
            p = scaler_params[str(f_idx)]
            normalized.append((val - p["min"]) / p["range"])
        window.append(normalized)

    X = torch.tensor([window], dtype=torch.float32)
    probs = []

    for seed in SEEDS:
        model_path = f"{MODEL_DIR}/lstm_model_{seed}.pt"
        if not os.path.exists(model_path):
            continue
        model = SentimentLSTM()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            prob = float(model(X).view(-1).item())
            probs.append(prob)

    if not probs:
        # Fall back to single model
        from services.lstm_predictor import predict
        return predict(ticker, recent_data)

    prob_up = float(np.mean(probs))
    std = float(np.std(probs))
    signal = "BUY" if prob_up > 0.53 else "SELL" if prob_up < 0.47 else "HOLD"
    confidence = max(prob_up, 1 - prob_up)

    return {
        "ticker": ticker,
        "signal": signal,
        "prob_up": round(prob_up, 3),
        "prob_down": round(1 - prob_up, 3),
        "confidence": round(confidence, 3),
        "confidence_pct": f"{confidence*100:.0f}%",
        "model_agreement": f"{(1 - std/0.5)*100:.0f}%",
        "models_used": len(probs),
    }


@router.get("/{ticker}")
async def get_prediction(ticker: str):
    ticker = ticker.upper()
    price_history = influx.query_price_history(ticker, hours=12)
    sent_history = influx.query_sentiment_history(ticker, hours=12)
    sent_map = {s["time"][:13]: s["value"] for s in sent_history}
    recent_data = [
        {
            "price": p["price"],
            "time": p["time"],
            "sentiment": sent_map.get(p["time"][:13], 50.0)
        }
        for p in price_history[-20:]
    ]
    return ensemble_predict(ticker, recent_data)


@router.get("/")
async def get_all_predictions():
    results = []
    for ticker in TICKERS:
        price_history = influx.query_price_history(ticker, hours=12)
        sent_history = influx.query_sentiment_history(ticker, hours=12)
        sent_map = {s["time"][:13]: s["value"] for s in sent_history}
        recent_data = [
            {
                "price": p["price"],
                "time": p["time"],
                "sentiment": sent_map.get(p["time"][:13], 50.0)
            }
            for p in price_history[-20:]
        ]
        result = ensemble_predict(ticker, recent_data)
        result["ticker"] = ticker
        results.append(result)
    return results