import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Cross-sector generalization test.
Train on tech + financials, test on healthcare + energy + consumer.
If accuracy stays above 53% on unseen sectors, the model learned
generalizable market patterns, not sector-specific memorization.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from services.lstm_predictor import (
    build_sequences,
    fetch_training_data,
    SentimentLSTM,
    SEQUENCE_LEN, FEATURES, LR
)

TRAIN_TICKERS = [
    "AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META",
    "AMD", "INTC", "QCOM", "AVGO", "MU", "SMCI", "ARM",
    "JPM", "BAC", "GS", "MS", "V", "MA",
    "SPY", "QQQ", "IWM", "XLK", "XLF",
]

TEST_TICKERS = [
    "JNJ", "PFE", "UNH", "MRNA", "ABBV", "LLY",
    "XOM", "CVX", "SLB", "COP",
    "WMT", "HD", "NKE", "MCD", "SBUX", "COST",
]

SEED = 123
EPOCHS = 150


def build_sequences_for_tickers(data, tickers):
    subset = [r for r in data if r["ticker"] in tickers]
    if not subset:
        return np.empty((0, SEQUENCE_LEN, FEATURES), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return build_sequences(subset)


def run():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Fetching data...")
    data = fetch_training_data()

    print(f"\nBuilding TRAIN sequences from {len(TRAIN_TICKERS)} tickers:")
    print(f"  {', '.join(TRAIN_TICKERS)}")
    X_train, y_train = build_sequences_for_tickers(data, TRAIN_TICKERS)

    print(f"\nBuilding TEST sequences from {len(TEST_TICKERS)} tickers (UNSEEN):")
    print(f"  {', '.join(TEST_TICKERS)}")
    X_test, y_test = build_sequences_for_tickers(data, TEST_TICKERS)

    print(f"\nTrain: {len(X_train)} sequences | Test: {len(X_test)} sequences")
    print(f"Test class balance: {y_test.mean():.1%} up, {1-y_test.mean():.1%} down")

    # Balance training set
    up_idx = np.where(y_train == 1)[0]
    down_idx = np.where(y_train == 0)[0]
    min_count = min(len(up_idx), len(down_idx))
    idx = np.concatenate([
        np.random.choice(up_idx, min_count, replace=False),
        np.random.choice(down_idx, min_count, replace=False)
    ])
    np.random.shuffle(idx)
    X_train = X_train[idx]
    y_train = y_train[idx]
    print(f"Balanced train: {len(X_train)} sequences")

    # Normalize using train statistics only
    X_train_clipped = np.clip(X_train, -5, 5)
    X_test_clipped = np.clip(X_test, -5, 5)

    X_train_t = torch.tensor(X_train_clipped, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_clipped, dtype=torch.float32)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SentimentLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
    criterion = nn.BCELoss()

    print("\nTraining on tech + financials sectors...")
    best_train_acc = 0

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
        scheduler.step()

        if (epoch + 1) % 30 == 0:
            model.eval()
            with torch.no_grad():
                train_pred = (model(X_train_t).view(-1) > 0.5).float()
                train_acc = accuracy_score(y_train_t.numpy(), train_pred.numpy())
                test_pred = (model(X_test_t).view(-1) > 0.5).float()
                test_acc = accuracy_score(y_test, test_pred.numpy())
                print(f"  Epoch {epoch+1}/{EPOCHS} — train: {train_acc:.1%} — unseen sectors: {test_acc:.1%}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_pred = (model(X_train_t).view(-1) > 0.5).float()
        train_acc = accuracy_score(y_train_t.numpy(), train_pred.numpy())
        test_pred = (model(X_test_t).view(-1) > 0.5).float()
        test_acc = accuracy_score(y_test, test_pred.numpy())

    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Train accuracy (tech + financials) : {train_acc:.1%}")
    print(f"Test accuracy  (healthcare + energy + consumer) : {test_acc:.1%}")
    print(f"Generalization gap : {(train_acc - test_acc)*100:.1f} points")

    if test_acc >= 0.55:
        print("\nRESULT: STRONG — model learned generalizable cross-sector patterns")
    elif test_acc >= 0.53:
        print("\nRESULT: MODERATE — some cross-sector generalization")
    elif test_acc >= 0.51:
        print("\nRESULT: WEAK — limited cross-sector generalization")
    else:
        print("\nRESULT: POOR — model likely memorized sector-specific patterns")
    print(f"{'='*50}")


if __name__ == "__main__":
    run()