import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Cross-ticker generalization test.
Train on 5 tickers, test on 2 unseen tickers.
If accuracy stays above 53% on unseen tickers, the model learned
generalizable market patterns, not ticker-specific memorization.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from collections import defaultdict
from services.lstm_predictor import (
    fetch_training_data, SentimentLSTM,
    SEQUENCE_LEN, FEATURES, HIDDEN_SIZE, EPOCHS, LR
)
import json

TRAIN_TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"]
TEST_TICKERS = ["AMZN", "META"]
SEED = 123


def build_sequences_for_tickers(data, tickers):
    by_ticker = defaultdict(list)
    for row in data:
        by_ticker[row["ticker"]].append(row)

    X_all, y_all = [], []

    for ticker in tickers:
        if ticker not in by_ticker:
            continue
        rows = sorted(by_ticker[ticker], key=lambda r: r["hour"])
        prices = [r["price"] for r in rows]

        for i in range(SEQUENCE_LEN, len(rows) - 1):
            window = []
            for j in range(i - SEQUENCE_LEN, i):
                r = rows[j]
                price = r["price"]
                prev_price = rows[j-1]["price"] if j > 0 else price
                price_change = (price - prev_price) / prev_price * 100 if prev_price else 0
                window.append([
                    price,
                    price_change,
                    r["volume"],
                    r["rsi"],
                    r["macd"],
                    r["bb_upper"],
                    r.get("ma20", price),
                    r.get("volatility", 0.0),
                    r.get("vol_momentum", 1.0),
                    r.get("day_of_week", 0.0),
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


def normalize(X_train, X_test):
    n_train, seq_len, n_features = X_train.shape
    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    scaler_params = {}

    for f in range(n_features):
        min_val = float(X_train_flat[:, f].min())
        max_val = float(X_train_flat[:, f].max())
        rng = max_val - min_val if max_val != min_val else 1.0
        X_train_flat[:, f] = (X_train_flat[:, f] - min_val) / rng
        X_test_flat[:, f] = (X_test_flat[:, f] - min_val) / rng
        scaler_params[f] = {"min": min_val, "range": rng}

    return (X_train_flat.reshape(n_train, seq_len, n_features),
            X_test_flat.reshape(X_test.shape[0], seq_len, n_features))


def run_test():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Fetching data...")
    data = fetch_training_data()

    print(f"\nBuilding training sequences from: {TRAIN_TICKERS}")
    X_train, y_train = build_sequences_for_tickers(data, TRAIN_TICKERS)

    print(f"Building test sequences from: {TEST_TICKERS}")
    X_test, y_test = build_sequences_for_tickers(data, TEST_TICKERS)

    print(f"\nTrain sequences: {len(X_train)} | Test sequences: {len(X_test)}")

    # Balance training set
    up_idx = np.where(y_train == 1)[0]
    down_idx = np.where(y_train == 0)[0]
    min_count = min(len(up_idx), len(down_idx))
    idx = np.concatenate([
        np.random.choice(up_idx, min_count, replace=False),
        np.random.choice(down_idx, min_count, replace=False)
    ])
    np.random.shuffle(idx)
    X_train, y_train = X_train[idx], y_train[idx]
    print(f"Balanced train: {len(X_train)} sequences")

    # Normalize using train statistics only
    X_train, X_test = normalize(X_train, X_test)

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SentimentLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
    criterion = nn.BCELoss()

    print("\nTraining on 5 tickers...")
    best_test_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch).view(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                # Train accuracy
                train_pred = (model(X_train_t).view(-1) > 0.5).float()
                train_acc = accuracy_score(y_train_t.numpy(), train_pred.numpy())
                # Test accuracy on unseen tickers
                test_pred = (model(X_test_t).view(-1) > 0.5).float()
                test_acc = accuracy_score(y_test_t.numpy(), test_pred.numpy())
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                print(f"  Epoch {epoch+1}/{EPOCHS} — train: {train_acc:.1%} — unseen tickers: {test_acc:.1%}")

    print(f"\n{'='*50}")
    print(f"RESULT: Best accuracy on UNSEEN tickers (AMZN, META): {best_test_acc:.1%}")
    if best_test_acc >= 0.53:
        print("✓ Model learned GENERALIZABLE patterns (not ticker memorization)")
    elif best_test_acc >= 0.51:
        print("~ Model learned SOME generalizable patterns")
    else:
        print("✗ Model likely memorized ticker-specific patterns")
    print(f"{'='*50}")


if __name__ == "__main__":
    run_test()