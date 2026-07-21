import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import json
import pandas as pd
import ta

MODEL_DIR = "/app/models"
MODEL_PATH = f"{MODEL_DIR}/lstm_model.pt"
SCALER_PATH = f"{MODEL_DIR}/scaler.json"

SEQUENCE_LEN=16
FEATURES=20
HIDDEN_SIZE=128
NUM_LAYERS=2
DROPOUT=0.3
EPOCHS=300
LR=0.001


class SentimentLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
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
    bb_width = (bb_upper - bb_lower) / close
    ma20 = close.rolling(window=20).mean()
    ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    atr = ta.volatility.AverageTrueRange(close, close, close, window=14).average_true_range()
    volatility = close.rolling(window=5).std()
    vol_ma5 = volume.rolling(window=5).mean()
    vol_momentum = volume / vol_ma5.replace(0, 1)
    day_of_week = pd.Series([0.0] * len(prices))
    adx = ta.trend.ADXIndicator(close, close, close, window=14).adx()
    obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    stoch = ta.momentum.StochasticOscillator(close, close, close, window=14).stoch()
    williams_r = ta.momentum.WilliamsRIndicator(close, close, close, lbp=14).williams_r()
    cci = ta.trend.CCIIndicator(close, close, close, window=20).cci()
    vwap = (close * volume).cumsum() / volume.cumsum()
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
        "volatility": volatility.fillna(0).tolist(),
        "vol_momentum": vol_momentum.fillna(1).tolist(),
        "day_of_week": day_of_week.tolist(),
        "adx": adx.fillna(25).tolist(),
        "obv": obv.fillna(0).tolist(),
        "stoch": stoch.fillna(50).tolist(),
        "williams_r": williams_r.fillna(-50).tolist(),
        "cci": cci.fillna(0).tolist(),
        "vwap": vwap.fillna(close).tolist(),
    }


def fetch_training_data():
    from influxdb_client import InfluxDBClient
    from config import settings
    client = InfluxDBClient(url=settings.influx_url, token=settings.influx_token, org=settings.influx_org)
    query_api = client.query_api()
    tickers = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]
    all_data = []

    for ticker in tickers:
        sent_flux = f'from(bucket: "sentiment_scores") |> range(start: -2y) |> filter(fn: (r) => r._measurement == "sentiment" and r.ticker == "{ticker}" and r._field == "composite") |> aggregateWindow(every: 1d, fn: mean, createEmpty: false) |> sort(columns: ["_time"])'
        price_flux = f'from(bucket: "sentiment_scores") |> range(start: -2y) |> filter(fn: (r) => r._measurement == "prices_daily" and r.ticker == "{ticker}" and r._field == "close") |> aggregateWindow(every: 1d, fn: mean, createEmpty: false) |> sort(columns: ["_time"])'
        volume_flux = f'from(bucket: "sentiment_scores") |> range(start: -2y) |> filter(fn: (r) => r._measurement == "prices_daily" and r.ticker == "{ticker}" and r._field == "volume") |> aggregateWindow(every: 1d, fn: mean, createEmpty: false) |> sort(columns: ["_time"])'
        try:
            sent_map = {}
            for table in query_api.query(sent_flux):
                for record in table.records:
                    hour = record.get_time().strftime("%Y-%m-%dT%H:00")
                    sent_map[hour] = record.get_value()

            price_map = {}
            for table in query_api.query(price_flux):
                for record in table.records:
                    hour = record.get_time().strftime("%Y-%m-%dT%H:00")
                    price_map[hour] = record.get_value()

            volume_map = {}
            for table in query_api.query(volume_flux):
                for record in table.records:
                    hour = record.get_time().strftime("%Y-%m-%dT%H:00")
                    volume_map[hour] = record.get_value()

            hours = sorted(price_map.keys())
            prices_list = [price_map[h] for h in hours]
            volumes_list = [volume_map.get(h, 0.0) for h in hours]

            if len(prices_list) >= 20:
                indicators = compute_indicators(prices_list, volumes_list, hours)
            else:
                indicators = {"rsi": [50.0]*len(hours), "macd": [0.0]*len(hours),
                             "bb_upper": prices_list, "bb_lower": prices_list}

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
                    "volatility": indicators["volatility"][idx],
                    "vol_momentum": indicators["vol_momentum"][idx],
                    "day_of_week": indicators["day_of_week"][idx],
                    "adx": indicators["adx"][idx],
                    "obv": indicators["obv"][idx],
                    "stoch": indicators["stoch"][idx],
                    "williams_r": indicators["williams_r"][idx],
                    "cci": indicators["cci"][idx],
                    "vwap": indicators["vwap"][idx],
                })
            print(f"  {ticker}: {len(price_map)} price pts, {len(sent_map)} sentiment pts")

        except Exception as e:
            print(f"  Error {ticker}: {e}")

    client.close()
    return all_data


def build_sequences(data):
    from collections import defaultdict
    by_ticker = defaultdict(list)
    for row in data:
        by_ticker[row["ticker"]].append(row)

    X_all, y_all = [], []

    for ticker, rows in by_ticker.items():
        rows = sorted(rows, key=lambda r: r["hour"])
        prices = [r["price"] for r in rows]

        for i in range(SEQUENCE_LEN, len(rows) - 1):
            window = []
            for j in range(i - SEQUENCE_LEN, i):
                r = rows[j]
                price = r["price"]
                prev_price = rows[j-1]["price"] if j > 0 else price
                price_change = (price - prev_price) / prev_price * 100 if prev_price else 0
                window.append([
                    r["sentiment"],
                    price,
                    price_change,
                    r["volume"],
                    r["rsi"],
                    r["macd"],
                    r["bb_upper"],
                    r.get("bb_lower", price),
                    r.get("bb_width", 0.0),
                    r.get("ma20", price),
                    r.get("ema20", price),
                    r.get("ema50", price),
                    r.get("atr", 0.0),
                    r.get("vol_momentum", 1.0),
                    r.get("adx", 25.0),
                    r.get("obv", 0.0),
                    r.get("stoch", 50.0),
                    r.get("williams_r", -50.0),
                    r.get("cci", 0.0),
                    r.get("vwap", price),
                ])

            if i + 3 >= len(prices):
                continue
            pct_change = (prices[i + 3] - prices[i]) / prices[i] * 100
            if pct_change > 0.2:
                label = 1
            elif pct_change < -0.2:
                label = 0
            else:
                continue

            X_all.append(window)
            y_all.append(label)

    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.float32)


def train():
    torch.manual_seed(123)
    np.random.seed(123)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Fetching training data from InfluxDB...")
    data = fetch_training_data()
    print(f"Got {len(data)} hourly data points")
    if len(data) < SEQUENCE_LEN + 2:
        print("Not enough data.")
        return None

    X, y = build_sequences(data)
    up_idx = np.where(y == 1)[0]
    down_idx = np.where(y == 0)[0]
    min_count = min(len(up_idx), len(down_idx))
    up_idx = np.random.choice(up_idx, min_count, replace=False)
    down_idx = np.random.choice(down_idx, min_count, replace=False)
    balanced_idx = np.concatenate([up_idx, down_idx])
    np.random.shuffle(balanced_idx)
    X = X[balanced_idx]
    y = y[balanced_idx]
    print(f"Balanced to {len(X)} sequences (50/50)")
    print(f"Built {len(X)} training sequences")
    print(f"Class balance: {y.mean():.1%} up, {1-y.mean():.1%} down")

    n_samples, seq_len, n_features = X.shape
    X_flat = X.reshape(-1, n_features)
    scaler_params = {}
    for f in range(n_features):
        min_val = float(X_flat[:, f].min())
        max_val = float(X_flat[:, f].max())
        rng = max_val - min_val if max_val != min_val else 1.0
        X_flat[:, f] = (X_flat[:, f] - min_val) / rng
        scaler_params[f] = {"min": min_val, "range": rng}
    X = X_flat.reshape(n_samples, seq_len, n_features)

    with open(SCALER_PATH, "w") as f:
        json.dump(scaler_params, f)

    split = int(0.8 * len(X))
    X_train_np = X[:split]
    y_train_np = y[:split]
    X_val_np = X[split:]
    y_val_np = y[split:]

    up_idx = np.where(y_train_np == 1)[0]
    down_idx = np.where(y_train_np == 0)[0]
    min_count = min(len(up_idx), len(down_idx))
    up_idx = np.random.choice(up_idx, min_count, replace=False)
    down_idx = np.random.choice(down_idx, min_count, replace=False)
    balanced_idx = np.concatenate([up_idx, down_idx])
    np.random.shuffle(balanced_idx)
    X_train_np = X_train_np[balanced_idx]
    y_train_np = y_train_np[balanced_idx]

    X_train = torch.tensor(X_train_np)
    y_train = torch.tensor(y_train_np)
    X_val = torch.tensor(X_val_np)
    y_val = torch.tensor(y_val_np)
    print(f"Train: {len(X_train)} sequences | Val: {len(X_val)} sequences")

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SentimentLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    criterion = nn.BCELoss()

    print("Training LSTM with technical indicators...")
    best_acc = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch).view(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                train_pred = (model(X_train).view(-1) > 0.5).float()
                train_acc = accuracy_score(y_train.numpy(), train_pred.numpy())
                val_pred = (model(X_val).view(-1) > 0.5).float()
                val_acc = accuracy_score(y_val.numpy(), val_pred.numpy())
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_state = model.state_dict().copy()
                print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {total_loss/len(loader):.4f} — train: {train_acc:.1%} — val: {val_acc:.1%}")

    if best_state:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")
    print(f"Best validation accuracy: {best_acc:.1%}")
    return model, best_acc


def predict(ticker, recent_data):
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not trained yet", "signal": "unknown"}

    with open(SCALER_PATH) as f:
        scaler_params = json.load(f)

    model = SentimentLSTM()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    if len(recent_data) < SEQUENCE_LEN:
        return {"error": f"Need {SEQUENCE_LEN} hours of data", "signal": "unknown"}

    prices = [r["price"] for r in recent_data]
    volumes = [r.get("volume", 0.0) for r in recent_data]

    if len(prices) >= 20:
        indicators = compute_indicators(prices, volumes)
    else:
        indicators = {"rsi": [50.0]*len(prices), "macd": [0.0]*len(prices),
                     "bb_upper": prices, "bb_lower": prices}

    window = []
    rows = recent_data[-SEQUENCE_LEN:]
    offset = len(recent_data) - SEQUENCE_LEN

    for i, row in enumerate(rows):
        idx = offset + i
        price = row["price"]
        prev_price = recent_data[offset+i-1]["price"] if i > 0 else price
        price_change = (price - prev_price) / prev_price * 100 if prev_price else 0
        features = [
            row.get("sentiment", 50.0),
            price,
            price_change,
            row.get("volume", 0.0),
            indicators["rsi"][idx] if idx < len(indicators["rsi"]) else 50.0,
            indicators["macd"][idx] if idx < len(indicators["macd"]) else 0.0,
            indicators["bb_upper"][idx] if idx < len(indicators["bb_upper"]) else price,
            indicators["bb_lower"][idx] if idx < len(indicators["bb_lower"]) else price,
            indicators["bb_width"][idx] if idx < len(indicators["bb_width"]) else 0.0,
            indicators["ma20"][idx] if idx < len(indicators["ma20"]) else price,
            indicators["ema20"][idx] if idx < len(indicators["ema20"]) else price,
            indicators["ema50"][idx] if idx < len(indicators["ema50"]) else price,
            indicators["atr"][idx] if idx < len(indicators["atr"]) else 0.0,
            indicators["vol_momentum"][idx] if idx < len(indicators["vol_momentum"]) else 1.0,
            indicators["adx"][idx] if idx < len(indicators["adx"]) else 25.0,
            indicators["obv"][idx] if idx < len(indicators["obv"]) else 0.0,
            indicators["stoch"][idx] if idx < len(indicators["stoch"]) else 50.0,
            indicators["williams_r"][idx] if idx < len(indicators["williams_r"]) else -50.0,
            indicators["cci"][idx] if idx < len(indicators["cci"]) else 0.0,
            indicators["vwap"][idx] if idx < len(indicators["vwap"]) else price,
        ]
            
        normalized = []
        for f_idx, val in enumerate(features):
            p = scaler_params[str(f_idx)]
            normalized.append((val - p["min"]) / p["range"])
        window.append(normalized)

    X = torch.tensor([window], dtype=torch.float32)
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