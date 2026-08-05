import os

import numpy as np
import torch
from fastapi import APIRouter, Depends

from services.auth import get_current_user
from services.lstm_predictor import (
    ENSEMBLE_SEEDS,
    SCALER_PATH,
    SEQUENCE_LEN,
    SentimentLSTM,
    apply_scaler,
    build_raw_features,
    fetch_live_daily_context,
    load_scaler,
)
from tickers import TICKERS, validate_ticker

router = APIRouter(prefix="/api/predictions", tags=["predictions"])

MODEL_DIR = "/app/models"
SEEDS = ENSEMBLE_SEEDS


def ensemble_predict(ticker: str, recent_data: list) -> dict:
    if not os.path.exists(SCALER_PATH):
        return {"error": "Model not trained", "signal": "HOLD",
                "prob_up": 0.5, "prob_down": 0.5, "confidence": 0.5, "confidence_pct": "50%"}

    if len(recent_data) < SEQUENCE_LEN:
        return {"error": f"Need {SEQUENCE_LEN} data points", "signal": "HOLD",
                "prob_up": 0.5, "prob_down": 0.5, "confidence": 0.5, "confidence_pct": "50%"}

    window = build_raw_features(recent_data)
    if window is None:
        return {"error": f"Need {SEQUENCE_LEN} data points", "signal": "HOLD",
                "prob_up": 0.5, "prob_down": 0.5, "confidence": 0.5, "confidence_pct": "50%"}

    X = torch.tensor(apply_scaler(window, load_scaler())[None], dtype=torch.float32)
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


@router.get("/{ticker}")
async def get_prediction(ticker: str, user=Depends(get_current_user)):
    ticker = validate_ticker(ticker)
    recent_data = fetch_live_daily_context(ticker, days=90)
    result = ensemble_predict(ticker, recent_data)
    result["ticker"] = ticker
    result["window_granularity"] = "daily"
    return result


@router.get("/")
async def get_all_predictions(user=Depends(get_current_user)):
    results = []
    for ticker in TICKERS:
        recent_data = fetch_live_daily_context(ticker, days=90)
        result = ensemble_predict(ticker, recent_data)
        result["ticker"] = ticker
        result["window_granularity"] = "daily"
        results.append(result)
    return results
