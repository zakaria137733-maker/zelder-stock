"""Deployed-artifact serving, live daily context, and deployed-model evaluation.

Artifact path constants, module-level caches, and the exact code path behind
/api/predictions: raw features -> scaler -> ensemble forward. Owns everything
that reads the deployed model files; training (ml_training) writes the same
paths so a retrained artifact is served without code changes.
"""

import json
import logging
import os

import numpy as np
import torch

from services.ml_features import (
    ENSEMBLE_SEEDS,
    LOOKBACK,
    SEQUENCE_LEN,
    SentimentLSTM,
    apply_scaler,
    build_raw_features,
    classification_metrics,
    prediction_evidence,
    signal_from_prob,
)

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
MODEL_PATH = f"{MODEL_DIR}/lstm_model.pt"
SCALER_PATH = f"{MODEL_DIR}/scaler.json"
PRED_THRESHOLDS_PATH = f"{MODEL_DIR}/predict_thresholds.json"

# Module-level caches for the deployed artifacts. The serving path (/api/
# predictions) used to re-load every model + json per request; now each artifact
# is loaded once and reused until its file signature (mtime+size) changes, so a
# retrained/deployed artifact is still picked up without a restart.
_ENSEMBLE_CACHE: dict = {}
_SCALER_CACHE: dict = {}
_THRESHOLDS_CACHE: dict = {}


def _path_signature(paths: list[str]) -> tuple:
    """A cache key for a set of artifact files: (path, mtime_ns, size) per file.

    Missing files use None so a later-created artifact invalidates the cache.
    """
    sig = []
    for path in paths:
        try:
            st = os.stat(path)
            sig.append((path, st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append((path, None))
    return tuple(sig)


def load_scaler() -> dict:
    """Load the per-feature min/max scaler saved at train time. Returns {} if absent.

    Cached at module level and invalidated when scaler.json changes, so the
    serving path reads it once instead of on every request.
    """
    sig = _path_signature([SCALER_PATH])
    cached = _SCALER_CACHE.get(sig)
    if cached is not None:
        return cached
    if not os.path.exists(SCALER_PATH):
        scaler = {}
    else:
        try:
            with open(SCALER_PATH) as f:
                scaler = json.load(f)
        except Exception as e:
            logger.warning("Scaler load error: %s", e)
            scaler = {}
    _SCALER_CACHE[sig] = scaler
    return scaler


def load_predict_thresholds() -> dict:
    """Load per-ticker serving thresholds/gate written by scripts/eval_lstm_signal.py.

    Cached at module level and invalidated when predict_thresholds.json changes.
    """
    sig = _path_signature([PRED_THRESHOLDS_PATH])
    cached = _THRESHOLDS_CACHE.get(sig)
    if cached is not None:
        return cached
    if not os.path.exists(PRED_THRESHOLDS_PATH):
        thresholds = {}
    else:
        try:
            with open(PRED_THRESHOLDS_PATH) as f:
                thresholds = json.load(f)
        except Exception as e:
            logger.warning("Predict-thresholds load error: %s", e)
            thresholds = {}
    _THRESHOLDS_CACHE[sig] = thresholds
    return thresholds


def load_ensemble_models() -> list:
    """Load the deployed 5-seed ensemble (falls back to the single model).

    Cached at module level, invalidated when any checkpoint file changes, so the
    serving path does 5 torch.loads once per artifact generation instead of on
    every request.
    """
    paths = [f"{MODEL_DIR}/lstm_model_{seed}.pt" for seed in ENSEMBLE_SEEDS]
    paths.append(MODEL_PATH)
    sig = _path_signature(paths)
    cached = _ENSEMBLE_CACHE.get(sig)
    if cached is not None:
        return cached
    models = []
    for seed in ENSEMBLE_SEEDS:
        path = f"{MODEL_DIR}/lstm_model_{seed}.pt"
        if os.path.exists(path):
            try:
                m = SentimentLSTM()
                m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                m.eval()
                models.append(m)
            except Exception as e:
                logger.warning("Model %s load error: %s", seed, e)
    if not models and os.path.exists(MODEL_PATH):
        try:
            m = SentimentLSTM()
            m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
            m.eval()
            models.append(m)
        except Exception as e:
            logger.warning("Single model load error: %s", e)
    _ENSEMBLE_CACHE[sig] = models
    return models


def ensemble_forward(X_scaled: np.ndarray) -> np.ndarray | None:
    """Run scaled windows through the deployed ensemble. Returns avg prob per sample."""
    models = load_ensemble_models()
    if not models:
        return None
    X = torch.tensor(np.asarray(X_scaled, dtype=np.float32))
    probs = np.zeros(X.shape[0], dtype=np.float32)
    for m in models:
        with torch.no_grad():
            probs += m(X).view(-1).numpy()
    return probs / len(models)


def _daily_returns(close_map: dict) -> dict:
    rets = {}
    prev = None
    for day in sorted(close_map.keys()):
        cur = close_map[day]
        rets[day] = (cur - prev) / prev * 100 if prev else 0.0
        prev = cur
    return rets


def fetch_live_daily_context(ticker: str, days: int = LOOKBACK) -> list[dict]:
    """Build recent_data at DAILY granularity to match fetch_training_data().

    Training reads daily bars from prices_daily and daily market_index series.
    Live inference must use the same granularity (10 daily steps).
    Daily closes come from prices_daily overlaid with fresh daily aggregates of
    the `prices` measurement; market context (spy_ret/vix) comes from the
    real market_index series instead of hardcoded defaults.

    The default lookback is LOOKBACK bars so indicator computation at serving
    time covers the same trailing context training uses (build_sequences).
    """
    from config import settings
    from services.influx import get_client

    client = get_client()
    query_api = client.query_api()
    bucket = settings.influx_bucket

    def query_map(measurement, field, agg, tag=ticker, extra_filter=""):
        flux = (
            "from(bucket: bucket)"
            " |> range(start: duration(v: days))"
            f' |> filter(fn: (r) => r._measurement == measurement'
            f' and r.ticker == tag and r._field == field{extra_filter})'
            f' |> aggregateWindow(every: 1d, fn: {agg}, createEmpty: false)'
            ' |> sort(columns: ["_time"])'
        )
        params = {
            "bucket": bucket,
            "days": f"-{days}d",
            "measurement": measurement,
            "tag": tag,
            "field": field,
        }
        out = {}
        try:
            for table in query_api.query(flux, params=params):
                for record in table.records:
                    out[record.get_time().strftime("%Y-%m-%d")] = float(record.get_value())
        except Exception as e:
            logger.warning("Influx query error (%s/%s/%s): %s", measurement, field, tag, e)
        return out

    price_close = query_map("prices_daily", "close", "last")
    price_volume = query_map("prices_daily", "volume", "last")
    live_close = query_map("prices", "close", "last")
    live_volume = query_map("prices", "volume", "last")
    sent_map = query_map("sentiment", "composite", "mean", extra_filter=' and r.source != "demo"')

    price_close.update(live_close)
    price_volume.update(live_volume)

    spy_map = query_map("market_index", "close", "last", tag="SPY")
    vix_map = query_map("market_index", "close", "last", tag="VIX")
    spy_ret = _daily_returns(spy_map)

    rows = []
    for day in sorted(price_close.keys()):
        rows.append({
            "ticker": ticker,
            "time": day,
            "price": price_close[day],
            "volume": price_volume.get(day, 0.0),
            "sentiment": sent_map.get(day, 50.0),
            "_sentiment_present": day in sent_map,
            "spy_ret": spy_ret.get(day, 0.0),
            "_spy_present": day in spy_map,
            "vix": vix_map.get(day, 20.0),
            "_vix_present": day in vix_map,
        })
    return rows


def _hold_response(ticker=None, error=None):
    """The neutral envelope returned whenever a prediction can't be made.

    Centralizes the four places the old code duplicated the same HOLD payload so
    the fallback shape (signal, prob_up/down, confidence, model_agreement,
    models_used, optional ticker/error) stays consistent across every caller.
    """
    env = {
        "signal": "HOLD",
        "prob_up": 0.5,
        "prob_down": 0.5,
        "confidence": 0.5,
        "confidence_pct": "50%",
        "model_agreement": "N/A",
        "models_used": 0,
    }
    if ticker is not None:
        env["ticker"] = ticker
    if error is not None:
        env["error"] = error
    return env


def predict_ensemble(ticker: str, recent_data: list) -> dict:
    """The single serving path behind /api/predictions.

    Loads the deployed 5-seed ensemble (falling back to the single model),
    builds the raw feature window, and returns the response envelope used by
    the API. Replaces the old duplicate ensemble loop in routers/predictions.py.
    """
    if not os.path.exists(SCALER_PATH):
        return _hold_response(error="Model not trained")

    if len(recent_data) < SEQUENCE_LEN:
        return _hold_response(error=f"Need {SEQUENCE_LEN} data points")

    window = build_raw_features(recent_data)
    if window is None:
        return _hold_response(error=f"Need {SEQUENCE_LEN} data points")

    X = torch.tensor(apply_scaler(window, load_scaler())[None], dtype=torch.float32)
    probs = []
    for model in load_ensemble_models():
        with torch.no_grad():
            probs.append(float(model(X).view(-1).item()))

    if not probs:
        return _hold_response(ticker=ticker)

    prob_up = float(np.mean(probs))
    std = float(np.std(probs)) if len(probs) > 1 else 0.0
    thresholds = load_predict_thresholds()
    meta = thresholds.get(ticker)
    signal = signal_from_prob(prob_up, ticker, thresholds)
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
        "signal_gate": bool(meta and meta.get("gate")),
        "evidence": prediction_evidence(meta) if meta else None,
    }


def evaluate_deployed(X_raw_val, y_val):
    """Evaluate the ACTUAL deployed artifact (5-seed ensemble + scaler.json).

    Uses the exact serving pipeline: raw features -> scaler -> ensemble forward,
    the same code path behind /api/predictions. Returns (metrics_dict, probs) or
    None if no trained model exists.
    """
    scaler = load_scaler()
    X_scaled = apply_scaler(np.asarray(X_raw_val, dtype=np.float32), scaler)
    probs = ensemble_forward(X_scaled)
    if probs is None:
        return None
    return classification_metrics(y_val, probs), probs
