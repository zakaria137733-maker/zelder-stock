import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Permutation test — if shuffled labels give similar accuracy,
the model learned noise not signal.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from services.lstm_predictor import (
    LR,
    SentimentLSTM,
    build_sequences,
    fetch_training_data,
)

N_PERMUTATIONS = 10
EPOCHS = 60

def train_and_eval(X_train, y_train, X_val, y_val, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    up_idx = np.where(y_train == 1)[0]
    down_idx = np.where(y_train == 0)[0]
    min_count = min(len(up_idx), len(down_idx))
    idx = np.concatenate([
        np.random.choice(up_idx, min_count, replace=False),
        np.random.choice(down_idx, min_count, replace=False)
    ])
    np.random.shuffle(idx)
    X_b = np.clip(X_train[idx], -5, 5)
    y_b = y_train[idx]

    X_t = torch.tensor(X_b, dtype=torch.float32)
    y_t = torch.tensor(y_b, dtype=torch.float32)
    X_v = torch.tensor(np.clip(X_val, -5, 5), dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)
    model = SentimentLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    criterion = nn.BCELoss()

    for _epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch).view(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = (model(X_v).view(-1) > 0.5).float()
        return accuracy_score(y_val, pred.numpy())


def run():
    print("Fetching data...")
    data = fetch_training_data()
    X, y, times = build_sequences(data, return_times=True)
    order = np.argsort(np.asarray(times), kind="stable")
    X, y = X[order], y[order]

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Real accuracy
    print("Training on REAL labels...")
    real_acc = train_and_eval(X_train, y_train, X_val, y_val, seed=123)
    print(f"Real accuracy: {real_acc:.1%}")

    # Permuted accuracies
    print(f"\nTraining on {N_PERMUTATIONS} SHUFFLED label sets...")
    perm_accs = []
    for i in range(N_PERMUTATIONS):
        y_shuffled = y_train.copy()
        np.random.shuffle(y_shuffled)
        acc = train_and_eval(X_train, y_shuffled, X_val, y_val, seed=i)
        perm_accs.append(acc)
        print(f"  Permutation {i+1}: {acc:.1%}")

    mean_perm = np.mean(perm_accs)
    print(f"\nReal accuracy      : {real_acc:.1%}")
    print(f"Mean shuffled acc  : {mean_perm:.1%}")
    print(f"Signal above noise : {(real_acc - mean_perm)*100:.1f} points")

    p_value = sum(a >= real_acc for a in perm_accs) / N_PERMUTATIONS
    print(f"p-value            : {p_value:.2f}")

    if p_value < 0.05:
        print("\nRESULT: STATISTICALLY SIGNIFICANT — real signal detected")
    elif p_value < 0.10:
        print("\nRESULT: MARGINAL — weak signal, needs more data to confirm")
    else:
        print("\nRESULT: NOT SIGNIFICANT — could be noise")


if __name__ == "__main__":
    run()
