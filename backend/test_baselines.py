import numpy as np
from services.lstm_predictor import fetch_training_data, build_sequences

data = fetch_training_data()
X, y = build_sequences(data)

split = int(0.8 * len(X))
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

# Baseline 3: momentum (last sequence label)
momentum_pred = X[split:, -1, 1] > 0  # last price_change > 0
acc_momentum = (momentum_pred.astype(float) == y_val).mean()
print(f"Momentum baseline     : {acc_momentum:.1%}")

print(f"\nYour ensemble         : 65.1%")
print(f"Gap above best base   : {(0.651 - max(acc_up, acc_random, acc_momentum))*100:.1f} points")