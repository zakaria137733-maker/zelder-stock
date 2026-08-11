import numpy as np
import torch
from services.lstm_predictor import fetch_training_data, build_sequences, SentimentLSTM, SCALER_PATH, FEATURES, SEQUENCE_LEN, HIDDEN_SIZE, EPOCHS, LR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import torch.nn as nn
import json
import os

MODEL_DIR="/app/models"
SEEDS = [42,123,7,99,2024]

def train_single(seed,X_train,y_train,X_val,y_val):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SentimentLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
    criterion = nn.BCELoss()

    best_acc = 0
    best_state = None

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
                val_pred = (model(X_val).view(-1) > 0.5).float()
                val_acc = accuracy_score(y_val.numpy(), val_pred.numpy())
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, best_acc


def train_ensemble():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Fetching training data...")
    data = fetch_training_data()
    X, y = build_sequences(data)

    up_idx = np.where(y == 1)[0]
    down_idx = np.where(y == 0)[0]
    min_count = min(len(up_idx), len(down_idx))
    up_idx = np.random.choice(up_idx, min_count, replace=False)
    down_idx = np.random.choice(down_idx, min_count, replace=False)
    idx = np.concatenate([up_idx, down_idx])
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]

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

    split=int(0.8 * len(X))
    X_train=torch.tensor(X[:split])
    y_train=torch.tensor(y[:split])
    X_val=torch.tensor(X[split:])
    y_val=torch.tensor(y[split:])

    accs = []
    for i, seed in enumerate(SEEDS):
        print(f"\nTraining model {i+1}/5 (seed={seed})...")
        model, acc = train_single(seed, X_train, y_train, X_val, y_val)
        torch.save(model.state_dict(), f"{MODEL_DIR}/lstm_model_{seed}.pt")
        accs.append(acc)
        print(f"  Seed {seed}: {acc:.1%}")

    print("\nEnsemble validation...")
    preds = []
    for seed in SEEDS:
        m = SentimentLSTM()
        m.load_state_dict(torch.load(f"{MODEL_DIR}/lstm_model_{seed}.pt", map_location="cpu"))
        m.eval()
        with torch.no_grad():
            preds.append(m(X_val).view(-1).numpy())

    avg_pred=np.mean(preds, axis=0)
    ensemble_pred=(avg_pred > 0.5).astype(float)
    ensemble_acc=accuracy_score(y_val.numpy(), ensemble_pred)
    print(f"\nIndividual accuracies:{[f'{a:.1%}' for a in accs]}")
    print(f"Ensemble accuracy:{ensemble_acc:.1%}")
    return ensemble_acc


if __name__ == "__main__":
    train_ensemble()