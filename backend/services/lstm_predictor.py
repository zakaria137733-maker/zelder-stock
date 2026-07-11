import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import json

MODEL_DIR = "/app/models"
MODEL_PATH = f"{MODEL_DIR}/lstm_model.pt"
SCALER_PATH = f"{MODEL_DIR}/scaler.json"

SEQUENCE_LEN = 6
FEATURES = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 90
LR = 0.001


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
            nn.Linear(HIDDEN_SIZE, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden).squeeze(-1)


def fetch_training_data():
    from influxdb_client import InfluxDBClient
    from config import settings
    client = InfluxDBClient(url=settings.influx_url, token=settings.influx_token, org=settings.influx_org)
    query_api = client.query_api()
    tickers = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]
    all_data = []
    for ticker in tickers:
        sent_flux = f'from(bucket: "sentiment_scores") |> range(start: -60d) |> filter(fn: (r) => r._measurement == "sentiment" and r.ticker == "{ticker}" and r._field == "composite") |> aggregateWindow(every: 1h, fn: mean, createEmpty: false) |> sort(columns: ["_time"])'
        price_flux = f'from(bucket: "sentiment_scores") |> range(start: -60d) |> filter(fn: (r) => r._measurement == "prices" and r.ticker == "{ticker}" and r._field == "close") |> aggregateWindow(every: 1h, fn: mean, createEmpty: false) |> sort(columns: ["_time"])'
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
            for hour in sorted(price_map.keys()):
                all_data.append({
                    "ticker": ticker,
                    "hour": hour,
                    "sentiment": sent_map.get(hour, 50.0),
                    "price": price_map[hour],
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
        sentiments = [r["sentiment"] for r in rows]
        for i in range(SEQUENCE_LEN, len(rows) - 1):
            window = []
            for j in range(i - SEQUENCE_LEN, i):
                price = prices[j]
                prev_price = prices[j-1] if j > 0 else price
                price_change = (price - prev_price) / prev_price * 100 if prev_price else 0
                window.append([sentiments[j], price, price_change, 15.0])
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
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Fetching training data from InfluxDB...")
    data = fetch_training_data()
    print(f"Got {len(data)} hourly data points")
    if len(data) < SEQUENCE_LEN + 2:
        print("Not enough data.")
        return None
    X, y = build_sequences(data)
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
    X_train = torch.tensor(X[:split])
    y_train = torch.tensor(y[:split])
    X_val = torch.tensor(X[split:])
    y_val = torch.tensor(y[split:])
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = SentimentLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    print("Training LSTM...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch).view(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = (model(X_val).view(-1) > 0.5).float()
                val_acc = accuracy_score(y_val.numpy(), val_pred.numpy())
                print(f"  Epoch {epoch+1}/{EPOCHS} — loss: {total_loss/len(loader):.4f} — val_acc: {val_acc:.1%}")
    model.eval()
    with torch.no_grad():
        val_pred = (model(X_val).view(-1) > 0.5).float()
        final_acc = accuracy_score(y_val.numpy(), val_pred.numpy())
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")
    print(f"Final validation accuracy: {final_acc:.1%}")
    return model, final_acc


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
    window = []
    rows = recent_data[-SEQUENCE_LEN:]
    for i, row in enumerate(rows):
        price = row["price"]
        prev_price = rows[i-1]["price"] if i > 0 else price
        price_change = (price - prev_price) / prev_price * 100 if prev_price else 0
        features = [row["sentiment"], price, price_change, 15.0]
        normalized = []
        for f_idx, val in enumerate(features):
            p = scaler_params[str(f_idx)]
            normalized.append((val - p["min"]) / p["range"])
        window.append(normalized)
    X = torch.tensor([window], dtype=torch.float32)
    with torch.no_grad():
        prob_up = float(model(X).view(-1).item())
    signal = "BUY" if prob_up > 0.6 else "SELL" if prob_up < 0.4 else "HOLD"
    confidence = max(prob_up, 1 - prob_up)
    return {
        "ticker": ticker,
        "signal": signal,
        "prob_up": round(prob_up, 3),
        "prob_down": round(1 - prob_up, 3),
        "confidence": round(confidence, 3),
        "confidence_pct": f"{confidence*100:.0f}%",
    }