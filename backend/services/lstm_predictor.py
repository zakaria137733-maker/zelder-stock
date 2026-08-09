import json
import os

import numpy as np
import pandas as pd
import ta
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
MODEL_PATH = f"{MODEL_DIR}/lstm_model.pt"
SCALER_PATH = f"{MODEL_DIR}/scaler.json"
PRED_THRESHOLDS_PATH = f"{MODEL_DIR}/predict_thresholds.json"

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
EPOCHS = 160
LR = 0.001
# Shared training hyperparameters — single-model train() and the 5-seed ensemble
# use the SAME values so they never drift apart again.
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

ENSEMBLE_SEEDS = [42, 123, 7, 99, 2024]

# The 7 tickers the live collector and GDELT backfill actually have sentiment
# for. Training on anything else silently treats a missing sentiment series as
# a constant 50.0, which makes the model's only unique input carry no signal.
# Single source of truth lives in tickers.py.
from tickers import TICKERS  # noqa: E402

TRACKED_TICKERS = list(TICKERS)

MIN_SENTIMENT_COVERAGE = 0.5
MIN_MARKET_COVERAGE = 0.8


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


def fetch_training_data():
    """Fetch training rows for the sentiment-backed tickers only.

    Restricting to TRACKED_TICKERS means the sentiment feature is always backed
    by real data instead of silently defaulting to 50.0 (the 43 tickers without
    a sentiment series previously polluted the training set with a constant
    feature). Market context is joined from the market_index measurement.
    """
    from services.influx import get_influx_client

    client = get_influx_client()
    query_api = client.query_api()
    tickers = TRACKED_TICKERS
    all_data = []
    days_range = "-5y"
    fetch_errors = {}

    for ticker in tickers:
        sent_flux = f'from(bucket: "sentiment_scores") |> range(start: {days_range}) |> filter(fn: (r) => r._measurement == "sentiment" and r.ticker == "{ticker}" and r._field == "composite" and r.source != "demo") |> aggregateWindow(every: 1d, fn: mean, createEmpty: false) |> sort(columns: ["_time"])'
        price_flux = f'from(bucket: "sentiment_scores") |> range(start: {days_range}) |> filter(fn: (r) => r._measurement == "prices_daily" and r.ticker == "{ticker}" and r._field == "close") |> aggregateWindow(every: 1d, fn: mean, createEmpty: false) |> sort(columns: ["_time"])'
        volume_flux = f'from(bucket: "sentiment_scores") |> range(start: {days_range}) |> filter(fn: (r) => r._measurement == "prices_daily" and r.ticker == "{ticker}" and r._field == "volume") |> aggregateWindow(every: 1d, fn: mean, createEmpty: false) |> sort(columns: ["_time"])'

        try:
            sent_map = {}
            for table in query_api.query(sent_flux):
                for record in table.records:
                    day = record.get_time().strftime("%Y-%m-%d")
                    sent_map[day] = record.get_value()

            price_map = {}
            for table in query_api.query(price_flux):
                for record in table.records:
                    day = record.get_time().strftime("%Y-%m-%d")
                    price_map[day] = record.get_value()

            volume_map = {}
            for table in query_api.query(volume_flux):
                for record in table.records:
                    day = record.get_time().strftime("%Y-%m-%d")
                    volume_map[day] = record.get_value()

            hours = sorted(price_map.keys())
            prices_list = [price_map[h] for h in hours]
            volumes_list = [volume_map.get(h, 0.0) for h in hours]

            if len(prices_list) >= 50:
                indicators = compute_indicators(prices_list, volumes_list)
            else:
                n = len(hours)
                indicators = {k: [v] * n for k, v in [
                    ("rsi", 50.0), ("macd", 0.0), ("bb_upper", 0.0), ("bb_lower", 0.0),
                    ("bb_width", 0.0), ("ma20", 0.0), ("ema20", 0.0), ("ema50", 0.0),
                    ("atr", 0.0), ("vol_momentum", 1.0), ("adx", 25.0), ("obv", 0.0),
                    ("stoch", 50.0), ("williams_r", -50.0), ("cci", 0.0), ("vwap", 0.0),
                ]}

            # compute_indicators() arrays are aligned to the LAST LOOKBACK bars;
            # index them at rel = idx - base (never the raw row index — that used
            # to overflow once a ticker had more than LOOKBACK days, truncating
            # each ticker to its oldest 260 days and misaligning every indicator).
            base = max(0, len(hours) - LOOKBACK)
            for idx, hour in enumerate(hours):
                rel = idx - base
                price = price_map[hour]
                all_data.append({
                    "ticker": ticker,
                    "hour": hour,
                    "time": hour,
                    "sentiment": sent_map.get(hour, 50.0),
                    "_sentiment_present": hour in sent_map,
                    "price": price,
                    "volume": volume_map.get(hour, 0.0),
                    "rsi": _ind_value(indicators, "rsi", rel, 50.0),
                    "macd": _ind_value(indicators, "macd", rel, 0.0),
                    "bb_upper": _ind_value(indicators, "bb_upper", rel, price),
                    "bb_lower": _ind_value(indicators, "bb_lower", rel, price),
                    "bb_width": _ind_value(indicators, "bb_width", rel, 0.0),
                    "ma20": _ind_value(indicators, "ma20", rel, price),
                    "ema20": _ind_value(indicators, "ema20", rel, price),
                    "ema50": _ind_value(indicators, "ema50", rel, price),
                    "atr": _ind_value(indicators, "atr", rel, 0.0),
                    "vol_momentum": _ind_value(indicators, "vol_momentum", rel, 1.0),
                    "adx": _ind_value(indicators, "adx", rel, 25.0),
                    "obv": _ind_value(indicators, "obv", rel, 0.0),
                    "stoch": _ind_value(indicators, "stoch", rel, 50.0),
                    "williams_r": _ind_value(indicators, "williams_r", rel, -50.0),
                    "cci": _ind_value(indicators, "cci", rel, 0.0),
                    "vwap": _ind_value(indicators, "vwap", rel, price),
                    "spy_ret": 0.0,
                    "qqq_ret": 0.0,
                    "vix": 20.0,
                    "_spy_present": False,
                    "_vix_present": False,
                })

            print(f"  {ticker}: {len(price_map)} price pts, {len(sent_map)} sentiment pts")

        except Exception as e:
            fetch_errors[ticker] = str(e)

    if fetch_errors:
        client.close()
        raise RuntimeError(
            "Failed to fetch "
            + ", ".join(f"{t} ({m})" for t, m in fetch_errors.items())
            + " — refusing to train on partial data."
        )

    # Fetch market indices
    print("Fetching market indices...")
    spy_map, qqq_map, vix_map = {}, {}, {}
    index_queries = [
        (f'from(bucket: "sentiment_scores") |> range(start: {days_range}) |> filter(fn: (r) => r._measurement == "market_index" and r.ticker == "SPY" and r._field == "close") |> aggregateWindow(every: 1d, fn: mean, createEmpty: false) |> sort(columns: ["_time"])', spy_map, "SPY"),
        (f'from(bucket: "sentiment_scores") |> range(start: {days_range}) |> filter(fn: (r) => r._measurement == "market_index" and r.ticker == "QQQ" and r._field == "close") |> aggregateWindow(every: 1d, fn: mean, createEmpty: false) |> sort(columns: ["_time"])', qqq_map, "QQQ"),
        (f'from(bucket: "sentiment_scores") |> range(start: {days_range}) |> filter(fn: (r) => r._measurement == "market_index" and r.ticker == "VIX" and r._field == "close") |> aggregateWindow(every: 1d, fn: mean, createEmpty: false) |> sort(columns: ["_time"])', vix_map, "VIX"),
    ]
    index_errors = []
    for flux, target, name in index_queries:
        try:
            for table in query_api.query(flux):
                for record in table.records:
                    day = record.get_time().strftime("%Y-%m-%d")
                    target[day] = record.get_value()
        except Exception as e:
            index_errors.append(f"{name}: {e}")
            print(f"  {name} error: {e}")
    if index_errors:
        client.close()
        raise RuntimeError(
            "Market index queries failed — " + "; ".join(index_errors)
            + " — refusing to train on partial market context."
        )

    print(f"  SPY: {len(spy_map)} pts | QQQ: {len(qqq_map)} pts | VIX: {len(vix_map)} pts")

    spy_hours = sorted(spy_map.keys())
    qqq_hours = sorted(qqq_map.keys())

    # Precomputed {day: previous-day close} so the per-row loop is O(1) instead of
    # re-scanning the sorted day list for every row.
    def prev_close_map(day_map, hours):
        prev = {}
        for i, day in enumerate(hours):
            prev[day] = day_map[hours[i - 1]] if i > 0 else day_map[day]
        return prev

    spy_prev_map = prev_close_map(spy_map, spy_hours)
    qqq_prev_map = prev_close_map(qqq_map, qqq_hours)

    for row in all_data:
        hour = row["hour"]
        spy_price = spy_map.get(hour, 0)
        spy_prev = spy_prev_map.get(hour, spy_price)

        qqq_price = qqq_map.get(hour, 0)
        qqq_prev = qqq_prev_map.get(hour, qqq_price)

        row["spy_ret"] = (spy_price - spy_prev) / spy_prev * 100 if spy_prev else 0
        row["qqq_ret"] = (qqq_price - qqq_prev) / qqq_prev * 100 if qqq_prev else 0
        row["vix"] = vix_map.get(hour, 20.0)
        row["_spy_present"] = hour in spy_map
        row["_vix_present"] = hour in vix_map

    client.close()
    print_coverage_report(all_data)
    return all_data


def coverage_report(data) -> dict:
    """Per-ticker coverage of the features that can silently default to constants.

    Missing sentiment defaults to 50.0, missing market context to spy_ret=0.0 /
    vix=20.0. Both look like plausible real values, so a model trained or served
    on thin data will quietly learn/emit nonsense. Every fetch path should call
    this so gaps are loud, not silent.
    """
    from collections import defaultdict

    by = defaultdict(lambda: {"days": 0, "sentiment": 0, "spy": 0, "vix": 0})
    for row in data:
        t = by[row["ticker"]]
        t["days"] += 1
        if row.get("_sentiment_present"):
            t["sentiment"] += 1
        if row.get("_spy_present"):
            t["spy"] += 1
        if row.get("_vix_present"):
            t["vix"] += 1

    report = {}
    for ticker, counts in sorted(by.items()):
        days = counts["days"]
        report[ticker] = {
            "days": days,
            "sentiment": round(counts["sentiment"] / days, 3) if days else 0.0,
            "spy": round(counts["spy"] / days, 3) if days else 0.0,
            "vix": round(counts["vix"] / days, 3) if days else 0.0,
        }
    return report


def print_coverage_report(data) -> dict:
    """Print and return coverage_report(). Warns when floors are missed."""
    report = coverage_report(data)
    for ticker, cov in report.items():
        flags = []
        if cov["sentiment"] < MIN_SENTIMENT_COVERAGE:
            flags.append(f"sentiment {cov['sentiment']:.0%} < {MIN_SENTIMENT_COVERAGE:.0%}")
        if cov["spy"] < MIN_MARKET_COVERAGE:
            flags.append(f"SPY {cov['spy']:.0%} < {MIN_MARKET_COVERAGE:.0%}")
        if cov["vix"] < MIN_MARKET_COVERAGE:
            flags.append(f"VIX {cov['vix']:.0%} < {MIN_MARKET_COVERAGE:.0%}")
        status = "OK" if not flags else "WARN " + "; ".join(flags)
        print(f"  coverage {ticker}: {cov['days']}d  sentiment={cov['sentiment']:.0%}  "
              f"SPY={cov['spy']:.0%}  VIX={cov['vix']:.0%}  [{status}]")
    return report


def require_coverage(data) -> dict:
    """Hard gate: raise RuntimeError when coverage floors are missed.

    Missing sentiment defaults to 50.0, missing market context to spy_ret=0.0 /
    vix=20.0 — both look like plausible real values, so a model trained on thin
    data quietly learns/emits nonsense. The coverage report only *warns*; this
    is what makes training stop instead of producing a confident-looking model
    with no signal. Returns the report when coverage is acceptable.
    """
    report = coverage_report(data)
    failures = []
    for ticker, cov in report.items():
        if cov["sentiment"] < MIN_SENTIMENT_COVERAGE:
            failures.append(f"{ticker}: sentiment {cov['sentiment']:.0%} < {MIN_SENTIMENT_COVERAGE:.0%}")
        if cov["spy"] < MIN_MARKET_COVERAGE:
            failures.append(f"{ticker}: SPY {cov['spy']:.0%} < {MIN_MARKET_COVERAGE:.0%}")
        if cov["vix"] < MIN_MARKET_COVERAGE:
            failures.append(f"{ticker}: VIX {cov['vix']:.0%} < {MIN_MARKET_COVERAGE:.0%}")
    if failures:
        raise RuntimeError(
            "Training data coverage is below the minimum floors — refusing to train a model on "
            "constant/missing features. Backfill the data first and retry:\n"
            "    docker-compose exec api python scripts/backfill_sentiment.py\n"
            "    docker-compose exec api python scripts/fetch_historical.py --period 5y\n"
            + "\n".join("  " + f for f in failures)
        )
    return report


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


def load_scaler() -> dict:
    """Load the per-feature min/max scaler saved at train time. Returns {} if absent."""
    if not os.path.exists(SCALER_PATH):
        return {}
    try:
        with open(SCALER_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"Scaler load error: {e}")
        return {}


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


def load_predict_thresholds() -> dict:
    """Load per-ticker serving thresholds/gate written by scripts/eval_lstm_signal.py."""
    if not os.path.exists(PRED_THRESHOLDS_PATH):
        return {}
    try:
        with open(PRED_THRESHOLDS_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"Predict-thresholds load error: {e}")
        return {}


def signal_from_prob(prob_up: float, ticker: str, thresholds: dict | None = None) -> str:
    """Decide the served signal for one ticker from walk-forward evidence.

    Without a passing gate there is no demonstrated edge, so we return
    NO_SIGNAL instead of a fabricated BUY/SELL. With a passing gate the Youden
    thresholds from the walk-forward OOF predictions replace the old 0.53/0.47.
    """
    if thresholds is None:
        thresholds = load_predict_thresholds()
    meta = thresholds.get(ticker)
    if not meta or not meta.get("gate"):
        return "NO_SIGNAL"
    if prob_up >= meta["buy_threshold"]:
        return "BUY"
    if prob_up <= meta["sell_threshold"]:
        return "SELL"
    return "NO_SIGNAL"


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
    from services.influx import get_influx_client

    client = get_influx_client()
    query_api = client.query_api()
    bucket = settings.influx_bucket
    try:
        def query_map(measurement, field, agg, tag=ticker, extra_filter=""):
            flux = (
                f'from(bucket: "{bucket}")'
                f' |> range(start: -{days}d)'
                f' |> filter(fn: (r) => r._measurement == "{measurement}"'
                f' and r.ticker == "{tag}" and r._field == "{field}"{extra_filter})'
                f' |> aggregateWindow(every: 1d, fn: {agg}, createEmpty: false)'
                f' |> sort(columns: ["_time"])'
            )
            out = {}
            try:
                for table in query_api.query(flux):
                    for record in table.records:
                        out[record.get_time().strftime("%Y-%m-%d")] = float(record.get_value())
            except Exception as e:
                print(f"Influx query error ({measurement}/{field}/{tag}): {e}")
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
    finally:
        client.close()


def load_ensemble_models() -> list:
    """Load the deployed 5-seed ensemble (falls back to the single model)."""
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
                print(f"Model {seed} load error: {e}")
    if not models and os.path.exists(MODEL_PATH):
        try:
            m = SentimentLSTM()
            m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
            m.eval()
            models.append(m)
        except Exception as e:
            print(f"Single model load error: {e}")
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


def train():
    torch.manual_seed(123)
    np.random.seed(123)

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Fetching training data from InfluxDB...")
    data = fetch_training_data()
    print(f"Got {len(data)} daily data points")
    print_coverage_report(data)
    require_coverage(data)

    if len(data) < SEQUENCE_LEN + 2:
        print("Not enough data.")
        return None

    X, y, times = build_sequences(data, return_times=True)
    print(f"Built {len(X)} training sequences")
    if len(X) == 0:
        print("No labeled windows — prices did not move beyond the threshold.")
        return None
    print(f"Class balance: {y.mean():.1%} up, {1-y.mean():.1%} down")

    order = np.argsort(np.asarray(times), kind="stable")
    X, y = X[order], y[order]
    n = len(X)
    # Three-way TEMPORAL split: train (oldest) / val (middle) / test (newest).
    # The test fold is never used for model selection or early stopping, so the
    # reported metrics measure real generalization instead of the val fold the
    # checkpoints were picked on.
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    X_train_raw, y_train_raw = X[:n_train], y[:n_train]
    X_val_raw, y_val_raw = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test_raw, y_test_raw = X[n_train + n_val:], y[n_train + n_val:]
    print(f"Temporal split (train / val / held-out test): "
          f"{len(X_train_raw)} / {len(X_val_raw)} / {len(X_test_raw)} windows")

    up_idx = np.where(y_train_raw == 1)[0]
    down_idx = np.where(y_train_raw == 0)[0]
    min_count = min(len(up_idx), len(down_idx))
    up_idx = np.random.choice(up_idx, min_count, replace=False)
    down_idx = np.random.choice(down_idx, min_count, replace=False)
    balanced_idx = np.concatenate([up_idx, down_idx])
    np.random.shuffle(balanced_idx)
    X_train_raw = X_train_raw[balanced_idx]
    y_train_raw = y_train_raw[balanced_idx]
    print(f"Balanced train: {len(X_train_raw)} | Val: {len(X_val_raw)} | Test: {len(X_test_raw)}")

    scaler_params = fit_scaler(X_train_raw)
    with open(SCALER_PATH, "w") as f:
        json.dump(scaler_params, f)
    X_train = torch.tensor(apply_scaler(X_train_raw, scaler_params), dtype=torch.float32)
    y_train = torch.tensor(y_train_raw, dtype=torch.float32)
    X_val = torch.tensor(apply_scaler(X_val_raw, scaler_params), dtype=torch.float32)
    y_val = torch.tensor(y_val_raw, dtype=torch.float32)
    X_test = torch.tensor(apply_scaler(X_test_raw, scaler_params), dtype=torch.float32)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SentimentLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
    criterion = nn.BCELoss()

    print("Training LSTM...")
    # Select on balanced accuracy, not raw accuracy: raw accuracy rewards
    # predicting the majority class on an imbalanced val fold (which is exactly
    # how the old runs ended up at the majority baseline).
    best_bal = 0.0
    best_state = None
    patience = 15
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch).view(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_prob = model(X_val).view(-1)
            val_pred = (val_prob > 0.5).float()
            val_bal = balanced_accuracy_score(y_val.numpy(), val_pred.numpy())

        if val_bal > best_bal:
            best_bal = val_bal
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                train_pred = (model(X_train).view(-1) > 0.5).float()
                train_acc = accuracy_score(y_train.numpy(), train_pred.numpy())
            print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {total_loss/len(loader):.4f} — train: {train_acc:.1%} — val: {val_bal:.1%}")

    if best_state:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")

    # Final metrics on the HELD-OUT test fold (never touched during selection).
    model.eval()
    with torch.no_grad():
        test_probs = model(X_test).view(-1).numpy()
    metrics = classification_metrics(y_test_raw, test_probs)
    metrics["persistence_baseline"] = persistence_baseline(X_test_raw, y_test_raw)
    print(f"Held-out test ({len(y_test_raw)} newest windows):")
    print(f"  accuracy={metrics['accuracy']:.1%} balanced={metrics['balanced_accuracy']:.1%} "
          f"auc={metrics['auc'] if metrics['auc'] is not None else 'N/A'} "
          f"majority_baseline={metrics['majority_baseline']:.1%} "
          f"persistence={metrics['persistence_baseline']:.1%}")
    if metrics["balanced_accuracy"] <= metrics["majority_baseline"]:
        print("  WARN: no evidence the model beats the majority baseline on held-out data.")
    return model, metrics


def predict_ensemble(ticker: str, recent_data: list) -> dict:
    """The single serving path behind /api/predictions.

    Loads the deployed 5-seed ensemble (falling back to the single model),
    builds the raw feature window, and returns the response envelope used by
    the API. Replaces the old duplicate ensemble loop in routers/predictions.py.
    """
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
    for model in load_ensemble_models():
        with torch.no_grad():
            probs.append(float(model(X).view(-1).item()))

    if not probs:
        return {"ticker": ticker, "signal": "HOLD", "prob_up": 0.5,
                "prob_down": 0.5, "confidence": 0.5, "confidence_pct": "50%",
                "model_agreement": "N/A", "models_used": 0}

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


def prediction_evidence(meta: dict) -> dict:
    """The walk-forward evidence behind a per-ticker signal, for API responses."""
    keys = [
        "n_windows", "lstm_acc", "momentum_acc", "majority_acc", "auc",
        "balanced_accuracy", "p_vs_momentum", "buy_threshold", "sell_threshold",
    ]
    return {k: meta[k] for k in keys if k in meta}
