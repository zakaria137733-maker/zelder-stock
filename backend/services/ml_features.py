"""Model architecture, feature construction, and label construction.

Model-agnostic pieces that both training (ml_training) and serving (ml_serving)
build on: the SentimentLSTM, the raw-feature window builders shared by train and
serve, the scaler helpers, and the metric helpers. Nothing here reads the
deployed artifacts or talks to InfluxDB, so training and serving can both depend
on it without a module cycle.
"""

import numpy as np
import pandas as pd
import ta
import torch.nn as nn

from tickers import TICKERS

SEQUENCE_LEN = 10
FEATURES = 12
HORIZON = 5
LABEL_THRESHOLD = 1.0
# Fixed indicator lookback shared by training and serving. Both build_sequences()
# and build_raw_features() compute indicators over the (at most) LOOKBACK bars
# ending at the window's final bar, so the deployed model always sees the same
# feature distribution at inference that it was trained on (no cumulative
# OBV/vwap drift from mismatched series lengths).
LOOKBACK = 260
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.5

ENSEMBLE_SEEDS = [42, 123, 7, 99, 2024]

# The 7 tickers the live collector and GDELT backfill actually have sentiment
# for. Training on anything else silently treats a missing sentiment series as
# a constant 50.0, which makes the model's only unique input carry no signal.
# Single source of truth lives in tickers.py.
TRACKED_TICKERS = list(TICKERS)


class SentimentLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=0.0,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden).squeeze(-1)


def compute_indicators(prices: list, volumes: list, lookback: int = LOOKBACK) -> dict:
    """Compute technical indicators on series.tail(lookback).

    Both build_sequences() and build_raw_features() call this with a series that
    is already truncated to the window's own LOOKBACK-bar context, so training
    and serving compute indicators over identical bars (including cumulative
    ones like OBV). The truncation here is a defensive no-op for those callers.

    The returned arrays are aligned to the LAST `lookback` bars; callers index
    them with ``idx - base`` where ``base = max(0, len(series) - lookback)``.
    """
    close = pd.Series(list(prices)[-lookback:])
    volume = pd.Series(list(volumes)[-lookback:])

    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close).macd()
    bb = ta.volatility.BollingerBands(close, window=20)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_width = (bb_upper - bb_lower) / close.replace(0, 1)
    ma20 = close.rolling(window=20).mean()
    ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    atr = ta.volatility.AverageTrueRange(close, close, close, window=14).average_true_range()
    vol_ma5 = volume.rolling(window=5).mean()
    vol_momentum = volume / vol_ma5.replace(0, 1)
    adx = ta.trend.ADXIndicator(close, close, close, window=14).adx()
    obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    stoch = ta.momentum.StochasticOscillator(close, close, close, window=14).stoch()
    williams_r = ta.momentum.WilliamsRIndicator(close, close, close, lbp=14).williams_r()
    cci = ta.trend.CCIIndicator(close, close, close, window=20).cci()
    vwap = (close * volume).cumsum() / volume.cumsum().replace(0, 1)

    return {
        "rsi": rsi.fillna(50).tolist(),
        "macd": macd.fillna(0).tolist(),
        "bb_upper": bb_upper.fillna(close).tolist(),
        "bb_lower": bb_lower.fillna(close).tolist(),
        "bb_width": bb_width.fillna(0).tolist(),
        "ma20": ma20.fillna(close).tolist(),
        "ema20": ema20.fillna(close).tolist(),
        "ema50": ema50.fillna(close).tolist(),
        "atr": atr.fillna(0).tolist(),
        "vol_momentum": vol_momentum.fillna(1).tolist(),
        "adx": adx.fillna(25).tolist(),
        "obv": obv.fillna(0).tolist(),
        "stoch": stoch.fillna(50).tolist(),
        "williams_r": williams_r.fillna(-50).tolist(),
        "cci": cci.fillna(0).tolist(),
        "vwap": vwap.fillna(close).tolist(),
    }


def _ind_value(indicators: dict, key: str, rel: int, default: float) -> float:
    """Resolve one indicator value at a relative offset, or `default`.

    compute_indicators() returns arrays aligned to the LAST LOOKBACK bars.
    Callers must index them with ``rel = idx - base`` where
    ``base = max(0, len(series) - LOOKBACK)``; bars before that context have no
    computed value and fall back to `default`. This is the same rule
    build_sequences()/build_raw_features() use, so every consumer stays aligned.
    """
    vals = indicators.get(key, [])
    return vals[rel] if 0 <= rel < len(vals) else default


def _default_indicators(n: int) -> dict:
    """Neutral indicator series used when there is too little data for real indicators."""
    return {
        "rsi": [50.0] * n,
        "macd": [0.0] * n,
        "bb_width": [0.0] * n,
        "adx": [25.0] * n,
        "obv": [0.0] * n,
        "stoch": [50.0] * n,
        "williams_r": [-50.0] * n,
        "cci": [0.0] * n,
    }


def _feature_row(row, prev_price, ind, eps=1e-8) -> list:
    """Build the 12 features for one bar. `ind` resolves a precomputed indicator
    by key with a default. This is the SINGLE source of truth for features so
    training (build_sequences) and serving (build_raw_features) can never drift."""
    price = row["price"]
    price_change = (price - prev_price) / prev_price * 100 if prev_price else 0
    return [
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


def _window_indicators(rows, i: int, lookback: int = LOOKBACK) -> dict | None:
    """Indicators for a window ending at bar `i`, computed on the (at most)
    LOOKBACK bars ending at `i`.

    This is exactly the slice serving computes on (fetch_live_daily_context
    returns ~LOOKBACK bars and build_raw_features() uses the same trailing
    context), so cumulative indicators like OBV match training exactly.
    """
    context = rows[max(0, i - lookback + 1): i + 1]
    prices = [r["price"] for r in context]
    volumes = [r.get("volume", 0.0) for r in context]
    if len(prices) < 28:
        return None
    try:
        return compute_indicators(prices, volumes)
    except Exception:
        return None


def build_sequences(data, horizon=HORIZON, threshold=LABEL_THRESHOLD, return_times=False):
    """Build training windows with the SAME convention as serving/eval.

    A window covers SEQUENCE_LEN bars ending at bar `i` (inclusive), and its
    label is the forward return from bar `i` to bar `i + horizon`, thresholded at
    ±`threshold`% (neutral windows dropped). This matches build_raw_features() +
    build_eval_sequences(), so train, eval, and /api/predictions all mean the
    same thing.

    When return_times=True a third element is returned: the end-bar timestamp
    of each window, used for time-ordered (leak-free) splits.
    """
    from collections import defaultdict
    by_ticker = defaultdict(list)
    for row in data:
        by_ticker[row["ticker"]].append(row)

    X_all, y_all, t_all = [], [], []

    for _ticker, rows in by_ticker.items():
        rows = sorted(rows, key=lambda r: r.get("time") or r["hour"])
        prices = [r["price"] for r in rows]

        for i in range(SEQUENCE_LEN, len(rows) - horizon):
            cur = prices[i]
            nxt = prices[i + horizon]
            pct = (nxt - cur) / cur * 100 if cur else 0
            if pct > threshold:
                label = 1
            elif pct < -threshold:
                label = 0
            else:
                continue

            indicators = _window_indicators(rows, i)
            nctx = min(i + 1, LOOKBACK)
            if indicators is None:
                indicators = _default_indicators(nctx)
            base = max(0, i - LOOKBACK + 1)

            start = i - SEQUENCE_LEN + 1
            window = []
            for pos, j in enumerate(range(start, i + 1)):
                row = rows[j]
                prev_price = rows[j - 1]["price"] if pos > 0 else row["price"]
                rel = j - base

                def ind(key, default, _rel=rel, _indicators=indicators):
                    vals = _indicators.get(key, [])
                    return vals[_rel] if 0 <= _rel < len(vals) else default

                window.append(_feature_row(row, prev_price, ind))

            X_all.append(window)
            y_all.append(label)
            t_all.append(rows[i].get("time") or rows[i]["hour"])

    X = np.array(X_all, dtype=np.float32) if X_all else np.empty((0, SEQUENCE_LEN, FEATURES), dtype=np.float32)
    y = np.array(y_all, dtype=np.float32) if y_all else np.empty((0,), dtype=np.float32)
    if return_times:
        return X, y, t_all
    return X, y


def build_raw_features(recent_data, indicators=None) -> np.ndarray | None:
    """Build the raw (pre-scaler) feature window for a single ticker's recent_data.

    Mirrors the per-step construction in build_sequences() exactly so inference
    sees the same input distribution the model was trained on. Indicators are
    computed over the trailing (at most) LOOKBACK bars — the same slice
    build_sequences() uses per window — so the served window matches training
    (including cumulative features like OBV). Returns None if there is not
    enough data.
    """
    if len(recent_data) < SEQUENCE_LEN:
        return None

    prices = [r["price"] for r in recent_data]
    volumes = [r.get("volume", 0.0) for r in recent_data]

    if indicators is None:
        if len(prices) >= 28:
            try:
                indicators = compute_indicators(prices, volumes)
            except Exception:
                indicators = None
        if indicators is None:
            indicators = _default_indicators(min(len(prices), LOOKBACK))

    base = max(0, len(recent_data) - LOOKBACK)
    rows = recent_data[-SEQUENCE_LEN:]
    offset = len(recent_data) - SEQUENCE_LEN
    window = []
    for i, row in enumerate(rows):
        idx = offset + i
        rel = idx - base
        prev_price = recent_data[offset + i - 1]["price"] if i > 0 else row["price"]

        def ind(key, default, _rel=rel):
            vals = indicators.get(key, [])
            return vals[_rel] if 0 <= _rel < len(vals) else default

        window.append(_feature_row(row, prev_price, ind))

    return np.array(window, dtype=np.float32)


def build_eval_sequences(rows, horizon=HORIZON, threshold=LABEL_THRESHOLD):
    """Build (X_raw, y) windows from daily rows for evaluating the deployed artifact.

    Uses the exact label convention from build_sequences(): the window ends at
    bar `i` and the label is the forward return from bar `i` over `horizon` bars,
    thresholded at ±`threshold`% (neutral skipped). Each window is built with
    build_raw_features() (the same path used by /api/predictions), so the
    evaluation measures the shipped model on the same inputs it serves.
    """
    X_raw, y = [], []
    for i in range(SEQUENCE_LEN, len(rows) - horizon):
        window = build_raw_features(rows[: i + 1])
        if window is None:
            continue
        cur = rows[i]["price"]
        nxt = rows[i + horizon]["price"]
        pct = (nxt - cur) / cur * 100 if cur else 0
        if pct > threshold:
            label = 1
        elif pct < -threshold:
            label = 0
        else:
            continue
        X_raw.append(window)
        y.append(label)
    if not X_raw:
        return np.empty((0, SEQUENCE_LEN, FEATURES), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.array(X_raw, dtype=np.float32), np.array(y, dtype=np.float32)


def fit_scaler(X_raw) -> dict:
    """Per-feature min/max scaler fitted on raw training windows.

    The single source of truth for scaler construction, shared by the single-model
    train() path and train_ensemble.py. Fit on the TRAIN fold only so the
    validation/serving fold's stats never leak into the transform.
    """
    X = np.asarray(X_raw, dtype=np.float32)
    n, _seq, n_features = X.shape
    flat = X.reshape(-1, n_features)
    params = {}
    for f in range(n_features):
        min_val = float(flat[:, f].min())
        max_val = float(flat[:, f].max())
        rng = max_val - min_val if max_val != min_val else 1.0
        params[f] = {"min": min_val, "range": rng}
    return params


def apply_scaler(window, scaler: dict) -> np.ndarray:
    """Apply the training-time per-feature min-max scaler to a (..., FEATURES) window.

    Mirrors train_ensemble.py: X = (X - min) / range per feature. Missing scaler
    (e.g. single-model path) falls back to identity so train/serve stay consistent.
    """
    window = np.asarray(window, dtype=np.float32)
    if scaler:
        for f in range(window.shape[-1]):
            params = scaler.get(str(f)) or scaler.get(f)
            if not params:
                continue
            rng = float(params.get("range", 1.0) or 1.0)
            mn = float(params.get("min", 0.0))
            window[..., f] = (window[..., f] - mn) / rng
    return np.clip(window, -5, 5)


def youden_threshold(y_true, prob_up) -> float:
    """ROC threshold maximizing Youden's J = TPR - FPR for `prob_up`.

    Used to turn continuous prob_up into BUY/SELL cutoffs instead of the old
    arbitrary 0.53/0.47 band.
    """
    from sklearn.metrics import roc_curve

    y = np.asarray(y_true)
    p = np.asarray(prob_up)
    if len(np.unique(y)) < 2 or len(y) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    best = int(np.argmax(j))
    return float(thr[best])


def signal_from_prob(prob_up: float, ticker: str, thresholds: dict | None = None) -> str:
    """Decide the served signal for one ticker from walk-forward evidence.

    Without a passing gate there is no demonstrated edge, so we return
    NO_SIGNAL instead of a fabricated BUY/SELL. With a passing gate the Youden
    thresholds from the walk-forward OOF predictions replace the old 0.53/0.47.
    """
    if thresholds is None:
        from services.ml_serving import load_predict_thresholds
        thresholds = load_predict_thresholds()
    meta = thresholds.get(ticker)
    if not meta or not meta.get("gate"):
        return "NO_SIGNAL"
    if prob_up >= meta["buy_threshold"]:
        return "BUY"
    if prob_up <= meta["sell_threshold"]:
        return "SELL"
    return "NO_SIGNAL"


def prediction_evidence(meta: dict) -> dict:
    """The walk-forward evidence behind a per-ticker signal, for API responses."""
    keys = [
        "n_windows", "lstm_acc", "momentum_acc", "majority_acc", "auc",
        "balanced_accuracy", "p_vs_momentum", "buy_threshold", "sell_threshold",
    ]
    return {k: meta[k] for k in keys if k in meta}


def classification_metrics(y_true, prob_up):
    """Reliable summary metrics on a (possibly imbalanced) fold.

    Raw accuracy alone is misleading on imbalanced data, so we also report
    balanced accuracy, ROC-AUC, and the majority-class baseline to compare
    against.
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    y = np.asarray(y_true).astype(float)
    p = np.asarray(prob_up).astype(float)
    preds = (p > 0.5).astype(float)
    baseline = max(float(y.mean()), 1.0 - float(y.mean()))
    return {
        "n": int(len(y)),
        "up_share": round(float(y.mean()), 4),
        "accuracy": round(float((preds == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, preds)), 4),
        "auc": round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) > 1 else None,
        "majority_baseline": round(baseline, 4),
    }


def persistence_baseline(X_raw, y) -> float:
    """A no-model baseline: predict up iff the window's last bar closed up.

    Feature 1 of each bar is the daily price change; the window's final bar
    gives the most recent signal a trader could act on at prediction time.
    """
    X = np.asarray(X_raw, dtype=np.float32)
    if len(X) == 0:
        return 0.0
    last_change = X[:, -1, 1]
    preds = (last_change >= 0).astype(float)
    return float(np.mean(preds == np.asarray(y)))
