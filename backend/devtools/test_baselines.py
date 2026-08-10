import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from services.lstm_predictor import (
    fetch_training_data,
    build_sequences,
    evaluate_deployed,
)

data = fetch_training_data()
X, y = build_sequences(data)

split = int(0.8 * len(X))
X_val = X[split:]
y_val = y[split:]

# Baseline 1: always predict UP
always_up = np.ones_like(y_val)
acc_up = (always_up == y_val).mean()
print(f"Always UP baseline    : {acc_up:.1%}")

# Baseline 2: random
np.random.seed(42)
random_pred = np.random.randint(0, 2, len(y_val)).astype(float)
acc_random = (random_pred == y_val).mean()
print(f"Random baseline       : {acc_random:.1%}")

# Baseline 3: momentum (last price_change in window > 0)
momentum_pred = X_val[:, -1, 1] > 0
acc_momentum = (momentum_pred.astype(float) == y_val).mean()
print(f"Momentum baseline     : {acc_momentum:.1%}")

# Deployed artifact: the actual 5-seed ensemble + scaler served behind /api/predictions
result = evaluate_deployed(X_val, y_val)
if result is None:
    print("\nYour ensemble         : N/A — no trained models found in /app/models")
    print("Train an ensemble first (train_ensemble.py) and re-run this script.")
else:
    metrics, _ = result
    acc_deployed = metrics["accuracy"]
    print(f"\nYour ensemble         : {acc_deployed:.1%}")
    print(f"Gap above best base   : {(acc_deployed - max(acc_up, acc_random, acc_momentum))*100:.1f} points")
