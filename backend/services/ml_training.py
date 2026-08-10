"""Training data fetching and model training.

Owns everything that touches InfluxDB for training plus the training loop and
its hyperparameters. Writes the artifact paths that ml_serving reads
(MODEL_PATH/SCALER_PATH), so training and serving stay symmetric without a
module cycle.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from services.ml_features import (
    LOOKBACK,
    SEQUENCE_LEN,
    TRACKED_TICKERS,
    SentimentLSTM,
    _ind_value,
    apply_scaler,
    build_sequences,
    classification_metrics,
    compute_indicators,
    fit_scaler,
    persistence_baseline,
)
from services.ml_serving import MODEL_DIR, MODEL_PATH, SCALER_PATH

# Shared training hyperparameters — single-model train() and the 5-seed ensemble
# use the SAME values so they never drift apart again.
EPOCHS = 160
LR = 0.001
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

MIN_SENTIMENT_COVERAGE = 0.5
MIN_MARKET_COVERAGE = 0.8


def _daily_range_flux(measurement, ticker, field, bucket, start="-5y", source_filter=False):
    """Build a daily-aggregated range query with Flux bind parameters.

    Returns (flux, params) for query_api.query(flux, params=params). Static
    clauses (source != "demo") stay inline; every interpolated value is a bind
    parameter, so tickers/buckets are never string-interpolated into Flux.
    """
    source = ' and r.source != "demo"' if source_filter else ""
    flux = (
        "from(bucket: bucket)"
        " |> range(start: duration(v: start))"
        f' |> filter(fn: (r) => r._measurement == measurement and r.ticker == ticker and r._field == field{source})'
        " |> aggregateWindow(every: 1d, fn: mean, createEmpty: false)"
        ' |> sort(columns: ["_time"])'
    )
    params = {
        "bucket": bucket,
        "start": start,
        "measurement": measurement,
        "ticker": ticker,
        "field": field,
    }
    return flux, params


def fetch_training_data():
    """Fetch training rows for the sentiment-backed tickers only.

    Restricting to TRACKED_TICKERS means the sentiment feature is always backed
    by real data instead of silently defaulting to 50.0 (the 43 tickers without
    a sentiment series previously polluted the training set with a constant
    feature). Market context is joined from the market_index measurement.
    """
    from config import settings
    from services.influx import get_influx_client

    client = get_influx_client()
    query_api = client.query_api()
    tickers = TRACKED_TICKERS
    all_data = []
    bucket = settings.influx_bucket
    fetch_errors = {}

    for ticker in tickers:
        sent_flux, sent_params = _daily_range_flux("sentiment", ticker, "composite", bucket, source_filter=True)
        price_flux, price_params = _daily_range_flux("prices_daily", ticker, "close", bucket)
        volume_flux, volume_params = _daily_range_flux("prices_daily", ticker, "volume", bucket)

        try:
            sent_map = {}
            for table in query_api.query(sent_flux, params=sent_params):
                for record in table.records:
                    day = record.get_time().strftime("%Y-%m-%d")
                    sent_map[day] = record.get_value()

            price_map = {}
            for table in query_api.query(price_flux, params=price_params):
                for record in table.records:
                    day = record.get_time().strftime("%Y-%m-%d")
                    price_map[day] = record.get_value()

            volume_map = {}
            for table in query_api.query(volume_flux, params=volume_params):
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
        (_daily_range_flux("market_index", "SPY", "close", bucket), spy_map, "SPY"),
        (_daily_range_flux("market_index", "QQQ", "close", bucket), qqq_map, "QQQ"),
        (_daily_range_flux("market_index", "VIX", "close", bucket), vix_map, "VIX"),
    ]
    index_errors = []
    for (flux, params), target, name in index_queries:
        try:
            for table in query_api.query(flux, params=params):
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
