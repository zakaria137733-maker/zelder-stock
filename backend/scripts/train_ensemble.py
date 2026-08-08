import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
from services.lstm_predictor import (
    EPOCHS,
    GRAD_CLIP,
    HORIZON,
    LABEL_THRESHOLD,
    LR,
    SCALER_PATH,
    SEQUENCE_LEN,
    WEIGHT_DECAY,
    SentimentLSTM,
    apply_scaler,
    build_sequences,
    classification_metrics,
    fetch_training_data,
    fit_scaler,
)
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
SEEDS = [42, 123, 7, 99, 2024]


def train_single(seed, X_train, y_train, X_val, y_val):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SentimentLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
    criterion = nn.BCELoss()

    best_acc = 0
    best_state = None
    patience = 15
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch).view(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = (model(X_val).view(-1) > 0.5).float()
            val_acc = accuracy_score(y_val.numpy(), val_pred.numpy())
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_acc


def train_ensemble(horizon=HORIZON, threshold=LABEL_THRESHOLD):
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)
    print(f"Fetching training data (horizon={horizon}d, label ±{threshold:g}%)...")
    data = fetch_training_data()
    X, y, times = build_sequences(data, horizon=horizon, threshold=threshold, return_times=True)
    if len(X) == 0:
        print("No labeled windows — nothing to train on.")
        return None
    print(f"Built {len(X)} windows (up {y.mean():.1%})")

    order = np.argsort(np.asarray(times), kind="stable")
    X, y = X[order], y[order]
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    print(f"Temporal split (train oldest / val newest): {len(X_train)} / {len(X_val)} windows")

    up_idx = np.where(y_train == 1)[0]
    down_idx = np.where(y_train == 0)[0]
    min_count = min(len(up_idx), len(down_idx))
    up_idx = np.random.choice(up_idx, min_count, replace=False)
    down_idx = np.random.choice(down_idx, min_count, replace=False)
    idx = np.concatenate([up_idx, down_idx])
    np.random.shuffle(idx)
    X_train, y_train = X_train[idx], y_train[idx]
    print(f"Balanced train: {len(X_train)} | Val: {len(X_val)}")

    scaler_params = fit_scaler(X_train)
    X_train_scaled = apply_scaler(X_train, scaler_params)
    X_val_scaled = apply_scaler(X_val, scaler_params)

    with open(SCALER_PATH, "w") as f:
        json.dump(scaler_params, f)
    print(f"Scaler written → {SCALER_PATH} (per-feature min/max fitted on train fold)")

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    accs = []
    for i, seed in enumerate(SEEDS):
        print(f"\nTraining model {i+1}/5 (seed={seed})...")
        model, acc = train_single(seed, X_train_t, y_train_t, X_val_t, y_val_t)
        torch.save(model.state_dict(), f"{MODEL_DIR}/lstm_model_{seed}.pt")
        accs.append(acc)
        print(f"  Seed {seed}: {acc:.1%}")

    print("\nEnsemble validation (newest held-out windows)...")
    preds = []
    for seed in SEEDS:
        m = SentimentLSTM()
        m.load_state_dict(torch.load(f"{MODEL_DIR}/lstm_model_{seed}.pt", map_location="cpu", weights_only=True))
        m.eval()
        with torch.no_grad():
            preds.append(m(X_val_t).view(-1).numpy())

    avg_prob = np.mean(preds, axis=0)
    metrics = classification_metrics(y_val.numpy(), avg_prob)
    print(f"  Individual accuracies: {[f'{a:.1%}' for a in accs]}")
    print(f"  Ensemble accuracy={metrics['accuracy']:.1%} "
          f"balanced={metrics['balanced_accuracy']:.1%} "
          f"auc={metrics['auc'] if metrics['auc'] is not None else 'N/A'} "
          f"baseline={metrics['majority_baseline']:.1%}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the 5-seed LSTM ensemble")
    parser.add_argument("--horizon", type=int, default=HORIZON,
                        help=f"forward-return horizon in bars (default {HORIZON})")
    parser.add_argument("--threshold", type=float, default=LABEL_THRESHOLD,
                        help=f"label move threshold in %% (default {LABEL_THRESHOLD:g})")
    args = parser.parse_args()
    train_ensemble(horizon=args.horizon, threshold=args.threshold)
