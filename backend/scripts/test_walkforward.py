import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Walk-forward validation — tests if accuracy is consistent
across different time periods, not just one lucky split.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from services.lstm_predictor import fetch_training_data, build_sequences, SentimentLSTM, FEATURES, SEQUENCE_LEN, LR, DROPOUT
from collections import defaultdict

N_WINDOWS = 5  # test across 5 different time periods
EPOCHS = 100

def train_window(X_train, y_train, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    up_idx = np.where(y_train == 1)[0]
    down_idx = np.where(y_train == 0)[0]
    min_count = min(len(up_idx), len(down_idx))
    if min_count < 10:
        return None
    idx = np.concatenate([
        np.random.choice(up_idx, min_count, replace=False),
        np.random.choice(down_idx, min_count, replace=False)
    ])
    np.random.shuffle(idx)
    X_b = np.clip(X_train[idx], -5, 5)
    y_b = y_train[idx]

    X_t = torch.tensor(X_b, dtype=torch.float32)
    y_t = torch.tensor(y_b, dtype=torch.float32)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SentimentLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    criterion = nn.BCELoss()

    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch).view(-1)
            loss = criterion(pred, y_batch)
            l1 = sum(p.abs().sum() for p in model.parameters())
            loss = loss + 1e-4 * l1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    return model


def run():
    print("Fetching data...")
    data = fetch_training_data()
    X, y, times = build_sequences(data, return_times=True)
    order = np.argsort(np.asarray(times), kind="stable")
    X, y = X[order], y[order]
    print(f"Total sequences: {len(X)}")

    window_size = len(X) // (N_WINDOWS + 1)
    results = []

    for i in range(N_WINDOWS):
        train_end = window_size * (i + 1)
        test_start = train_end
        test_end = test_start + window_size

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = np.clip(X[test_start:test_end], -5, 5)
        y_test = y[test_start:test_end]

        if len(X_test) < 50:
            continue

        model = train_window(X_train, y_train)
        if model is None:
            continue

        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            pred = (model(X_test_t).view(-1) > 0.5).float()
            acc = accuracy_score(y_test, pred.numpy())

        results.append(acc)
        print(f"  Window {i+1}: train={len(X_train)} test={len(X_test)} acc={acc:.1%}")

    print(f"\nMean accuracy  : {np.mean(results):.1%}")
    print(f"Std deviation  : {np.std(results):.1%}")
    print(f"Min accuracy   : {np.min(results):.1%}")
    print(f"Max accuracy   : {np.max(results):.1%}")

    if np.mean(results) > 0.55 and np.std(results) < 0.05:
        print("\nRESULT: SIGNAL — consistent accuracy across time periods")
    elif np.mean(results) > 0.53:
        print("\nRESULT: WEAK SIGNAL — some predictive power but inconsistent")
    else:
        print("\nRESULT: NOISE — accuracy not consistent across time periods")


if __name__ == "__main__":
    run()