import json
import os

import numpy as np
import pandas as pd
import ta
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
MODEL_PATH = f"{MODEL_DIR}/lstm_model.pt"
SCALER_PATH = f"{MODEL_DIR}/scaler.json"

SEQUENCE_LEN = 10
FEATURES = 12
HORIZON = 5
LABEL_THRESHOLD = 1.0
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.5
EPOCHS = 160
LR = 0.001

ENSEMBLE_SEEDS = [42, 123, 7, 99, 2024]


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


def compute_indicators(prices: list, volumes: list, dates: list = None) -> dict:
    close = pd.Series(prices)
    volume = pd.Series(volumes)

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

    day_of_week = pd.Series([0.0] * len(prices))
    if dates:
        try:
            day_of_week = pd.Series([pd.Timestamp(d).dayofweek / 4.0 for d in dates])
        except Exception:
            pass

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
        "day_of_week": day_of_week.tolist(),
    }


def fetch_training_data():
    from influxdb_client import InfluxDBClient

    from config import settings
    client = InfluxDBClient(url=settings.influx_url, token=settings.influx_token, org=settings.influx_org)
    query_api = client.query_api()
    tickers = [
    "AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META",
    "AMD", "INTC", "QCOM", "AVGO", "MU", "SMCI", "ARM",
    "JPM", "BAC", "GS", "MS", "V", "MA", "BRK-B",
    "JNJ", "PFE", "UNH", "MRNA", "ABBV", "LLY",
    "XOM", "CVX", "SLB", "COP",
    "WMT", "HD", "NKE", "MCD", "SBUX", "COST",
    "CRM", "SNOW", "PLTR", "NET", "DDOG", "NOW",
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE"]
    all_data = []

    for ticker in tickers:
        sent_flux = f'from(bucket: "sentiment_scores") |> range(start: -2y) |> filter(fn: (r) => r._measurement == "sentiment" and r.ticker == "{ticker}" and r._field == "composite") |> aggregateWindow(every: 1d, fn: mean, createEmpty: false) |> sort(columns: ["_time"])'
        price_flux = f'from(bucket: "sentiment_scores") |> range(start: -2y) |> filter(fn: (r) => r._measurement == "prices_daily" and r.ticker == "{ticker}" and r._field == "close") |> aggregateWindow(every: 1d, fn: mean, createEmpty: false) |> sort(columns: ["_time"])'
        volume_flux = f'from(bucket: "sentiment_scores") |> range(start: -2y) |> filter(fn: (r) => r._measurement == "prices_daily" and r.ticker == "{ticker}" and r._field == "volume") |> aggregateWindow(every: 1d, fn: mean, createEmpty: false) |> sort(columns: ["_time"])'

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
                indicators = compute_indicators(prices_list, volumes_list, hours)
            else:
                n = len(hours)
                indicators = {k: [v] * n for k, v in [
                    ("rsi", 50.0), ("macd", 0.0), ("bb_upper", 0.0), ("bb_lower", 0.0),
                    ("bb_width", 0.0), ("ma20", 0.0), ("ema20", 0.0), ("ema50", 0.0),
                    ("atr", 0.0), ("vol_momentum", 1.0), ("adx", 25.0), ("obv", 0.0),
                    ("stoch", 50.0), ("williams_r", -50.0), ("cci", 0.0), ("vwap", 0.0),
                    ("day_of_week", 0.0),
                ]}

            for idx, hour in enumerate(hours):
                all_data.append({
                    "ticker": ticker,
                    "hour": hour,
                    "sentiment": sent_map.get(hour, 50.0),
                    "price": price_map[hour],
                    "volume": volume_map.get(hour, 0.0),
                    "rsi": indicators["rsi"][idx],
                    "macd": indicators["macd"][idx],
                    "bb_upper": indicators["bb_upper"][idx],
                    "bb_lower": indicators["bb_lower"][idx],
                    "bb_width": indicators["bb_width"][idx],
                    "ma20": indicators["ma20"][idx],
                    "ema20": indicators["ema20"][idx],
                    "ema50": indicators["ema50"][idx],
                    "atr": indicators["atr"][idx],
                    "vol_momentum": indicators["vol_momentum"][idx],
                    "adx": indicators["adx"][idx],
                    "obv": indicators["obv"][idx],
                    "stoch": indicators["stoch"][idx],
                    "williams_r": indicators["williams_r"][idx],
                    "cci": indicators["cci"][idx],
                    "vwap": indicators["vwap"][idx],
                    "day_of_week": indicators["day_of_week"][idx],
                    "spy_ret": 0.0,
                    "qqq_ret": 0.0,
                    "vix": 20.0,
                })

            print(f"  {ticker}: {len(price_map)} price pts, {len(sent_map)} sentiment pts")

        except Exception as e:
            print(f"  Error {ticker}: {e}")

    # Fetch market indices
    print("Fetching market indices...")
    spy_map, qqq_map, vix_map = {}, {}, {}
    index_queries = [
        ('from(bucket: "sentiment_scores") |> range(start: -2y) |> filter(fn: (r) => r._measurement == "market_index" and r.ticker == "SPY" and r._field == "close") |> aggregateWindow(every: 1d, fn: mean, createEmpty: false) |> sort(columns: ["_time"])', spy_map, "SPY"),
        ('from(bucket: "sentiment_scores") |> range(start: -2y) |> filter(fn: (r) => r._measurement == "market_index" and r.ticker == "QQQ" and r._field == "close") |> aggregateWindow(every: 1d, fn: mean, createEmpty: false) |> sort(columns: ["_time"])', qqq_map, "QQQ"),
        ('from(bucket: "sentiment_scores") |> range(start: -2y) |> filter(fn: (r) => r._measurement == "market_index" and r.ticker == "VIX" and r._field == "close") |> aggregateWindow(every: 1d, fn: mean, createEmpty: false) |> sort(columns: ["_time"])', vix_map, "VIX"),
    ]
    for flux, target, name in index_queries:
        try:
            for table in query_api.query(flux):
                for record in table.records:
                    day = record.get_time().strftime("%Y-%m-%d")
                    target[day] = record.get_value()
        except Exception as e:
            print(f"  {name} error: {e}")

    print(f"  SPY: {len(spy_map)} pts | QQQ: {len(qqq_map)} pts | VIX: {len(vix_map)} pts")

    spy_hours = sorted(spy_map.keys())
    qqq_hours = sorted(qqq_map.keys())

    for row in all_data:
        hour = row["hour"]
        spy_price = spy_map.get(hour, 0)
        if hour in spy_map and spy_hours:
            spy_idx = spy_hours.index(hour)
            spy_prev = spy_map.get(spy_hours[spy_idx - 1], spy_price) if spy_idx > 0 else spy_price
        else:
            spy_prev = spy_price

        qqq_price = qqq_map.get(hour, 0)
        if hour in qqq_map and qqq_hours:
            qqq_idx = qqq_hours.index(hour)
            qqq_prev = qqq_map.get(qqq_hours[qqq_idx - 1], qqq_price) if qqq_idx > 0 else qqq_price
        else:
            qqq_prev = qqq_price

        row["spy_ret"] = (spy_price - spy_prev) / spy_prev * 100 if spy_prev else 0
        row["qqq_ret"] = (qqq_price - qqq_prev) / qqq_prev * 100 if qqq_prev else 0
        row["vix"] = vix_map.get(hour, 20.0)

    client.close()
    return all_data


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
        volumes = [r.get("volume", 0.0) for r in rows]
        dates = [r.get("time") or r["hour"] for r in rows]

        if len(prices) >= 28:
            try:
                indicators = compute_indicators(prices, volumes, dates)
            except Exception:
                indicators = None
        else:
            indicators = None
        if indicators is None:
            indicators = _default_indicators(len(prices))

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

            start = i - SEQUENCE_LEN + 1
            window = []
            for pos, j in enumerate(range(start, i + 1)):
                row = rows[j]
                prev_price = rows[j - 1]["price"] if pos > 0 else row["price"]

                def ind(key, default, _idx=j, _indicators=indicators):
                    vals = _indicators.get(key, [])
                    return vals[_idx] if _idx < len(vals) else default

                window.append(_feature_row(row, prev_price, ind))

            X_all.append(window)
            y_all.append(label)
            t_all.append(rows[i].get("time") or rows[i]["hour"])

    X = np.array(X_all, dtype=np.float32) if X_all else np.empty((0, SEQUENCE_LEN, FEATURES), dtype=np.float32)
    y = np.array(y_all, dtype=np.float32) if y_all else np.empty((0,), dtype=np.float32)
    if return_times:
        return X, y, t_all
    return X, y


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
    sees the same input distribution the model was trained on. Returns None if
    there is not enough data.
    """
    if len(recent_data) < SEQUENCE_LEN:
        return None

    prices = [r["price"] for r in recent_data]
    volumes = [r.get("volume", 0.0) for r in recent_data]
    dates = [r.get("time", "") for r in recent_data]

    if indicators is None:
        if len(prices) >= 28:
            try:
                indicators = compute_indicators(prices, volumes, dates)
            except Exception:
                indicators = None
        if indicators is None:
            indicators = _default_indicators(len(prices))

    rows = recent_data[-SEQUENCE_LEN:]
    offset = len(recent_data) - SEQUENCE_LEN
    window = []
    for i, row in enumerate(rows):
        idx = offset + i
        prev_price = recent_data[offset + i - 1]["price"] if i > 0 else row["price"]

        def ind(key, default, _idx=idx):
            vals = indicators.get(key, [])
            return vals[_idx] if _idx < len(vals) else default

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


def fetch_live_daily_context(ticker: str, days: int = 90) -> list[dict]:
    """Build recent_data at DAILY granularity to match fetch_training_data().

    Training reads daily bars from prices_daily and daily market_index series.
    Live inference must use the same granularity (10 daily steps).
    Daily closes come from prices_daily overlaid with fresh daily aggregates of
    Daily closes come from prices_daily overlaid with fresh daily aggregates of
    the `prices` measurement; market context (spy_ret/vix) comes from the
    real market_index series instead of hardcoded defaults.
    """
    from influxdb_client import InfluxDBClient

    from config import settings

    client = InfluxDBClient(url=settings.influx_url, token=settings.influx_token, org=settings.influx_org)
    query_api = client.query_api()
    bucket = settings.influx_bucket
    try:
        def query_map(measurement, field, agg, tag=ticker):
            flux = (
                f'from(bucket: "{bucket}")'
                f' |> range(start: -{days}d)'
                f' |> filter(fn: (r) => r._measurement == "{measurement}"'
                f' and r.ticker == "{tag}" and r._field == "{field}")'
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
        sent_map = query_map("sentiment", "composite", "mean")

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
                "spy_ret": spy_ret.get(day, 0.0),
                "vix": vix_map.get(day, 20.0),
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
                m.load_state_dict(torch.load(path, map_location="cpu"))
                m.eval()
                models.append(m)
            except Exception as e:
                print(f"Model {seed} load error: {e}")
    if not models and os.path.exists(MODEL_PATH):
        try:
            m = SentimentLSTM()
            m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
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


def train():
    torch.manual_seed(123)
    np.random.seed(123)

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Fetching training data from InfluxDB...")
    data = fetch_training_data()
    print(f"Got {len(data)} daily data points")

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
    split = int(0.8 * len(X))
    X_train_raw = X[:split]
    y_train_raw = y[:split]
    X_val_raw = X[split:]
    y_val_raw = y[split:]
    print(f"Temporal split (train oldest / val newest): {len(X_train_raw)} / {len(X_val_raw)} windows")

    up_idx = np.where(y_train_raw == 1)[0]
    down_idx = np.where(y_train_raw == 0)[0]
    min_count = min(len(up_idx), len(down_idx))
    up_idx = np.random.choice(up_idx, min_count, replace=False)
    down_idx = np.random.choice(down_idx, min_count, replace=False)
    balanced_idx = np.concatenate([up_idx, down_idx])
    np.random.shuffle(balanced_idx)
    X_train_raw = X_train_raw[balanced_idx]
    y_train_raw = y_train_raw[balanced_idx]
    print(f"Balanced train: {len(X_train_raw)} | Val: {len(X_val_raw)}")

    X_train_clipped = np.clip(X_train_raw, -5, 5)
    X_val_clipped = np.clip(X_val_raw, -5, 5)

    scaler_params = {str(f): {"min": 0.0, "range": 1.0} for f in range(FEATURES)}
    with open(SCALER_PATH, "w") as f:
        json.dump(scaler_params, f)

    X_train = torch.tensor(X_train_clipped, dtype=torch.float32)
    y_train = torch.tensor(y_train_raw, dtype=torch.float32)
    X_val = torch.tensor(X_val_clipped, dtype=torch.float32)
    y_val = torch.tensor(y_val_raw, dtype=torch.float32)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SentimentLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
    criterion = nn.BCELoss()

    print("Training LSTM...")
    best_acc = 0
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
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + 1e-4 * l1_norm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_prob = model(X_val).view(-1)
            val_pred = (val_prob > 0.5).float()
            val_acc = accuracy_score(y_val.numpy(), val_pred.numpy())

        if val_acc > best_acc:
            best_acc = val_acc
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
            print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {total_loss/len(loader):.4f} — train: {train_acc:.1%} — val: {val_acc:.1%}")

    if best_state:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")

    model.eval()
    with torch.no_grad():
        best_probs = model(X_val).view(-1).numpy()
    metrics = classification_metrics(y_val.numpy(), best_probs)
    print(f"Validation (newest {len(y_val)} windows):")
    print(f"  accuracy={metrics['accuracy']:.1%} balanced={metrics['balanced_accuracy']:.1%} "
          f"auc={metrics['auc'] if metrics['auc'] is not None else 'N/A'} baseline={metrics['majority_baseline']:.1%}")
    return model, metrics


def predict(ticker, recent_data):
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not trained yet", "signal": "unknown"}

    model = SentimentLSTM()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    if len(recent_data) < SEQUENCE_LEN:
        return {"error": f"Need {SEQUENCE_LEN} days of data", "signal": "unknown"}

    scaler = load_scaler()
    window = build_raw_features(recent_data)
    if window is None:
        return {"error": f"Need {SEQUENCE_LEN} days of data", "signal": "unknown"}
    X = torch.tensor(apply_scaler(window, scaler)[None], dtype=torch.float32)

    with torch.no_grad():
        prob_up = float(model(X).view(-1).item())

    signal = "BUY" if prob_up > 0.53 else "SELL" if prob_up < 0.47 else "HOLD"
    confidence = max(prob_up, 1 - prob_up)

    return {
        "ticker": ticker,
        "signal": signal,
        "prob_up": round(prob_up, 3),
        "prob_down": round(1 - prob_up, 3),
        "confidence": round(confidence, 3),
        "confidence_pct": f"{confidence*100:.0f}%",
    }
