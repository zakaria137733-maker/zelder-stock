import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import pickle
from collections import defaultdict

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from services.lstm_predictor import SEQUENCE_LEN, fetch_training_data


def build_flat_features(data):
    """Flatten LSTM sequences into single feature vectors for tree models."""
    by_ticker = defaultdict(list)
    for row in data:
        by_ticker[row["ticker"]].append(row)

    X_all, y_all = [], []

    for _ticker, rows in by_ticker.items():
        rows = sorted(rows, key=lambda r: r["hour"])
        prices = [r["price"] for r in rows]

        for i in range(SEQUENCE_LEN, len(rows) - 1):
            # Use last values of each indicator (not sequence)
            r = rows[i]
            price = r["price"]
            prev_price = rows[i-1]["price"]
            price_change = (price - prev_price) / prev_price * 100

            features = [
                r["sentiment"],
                price_change,
                r["rsi"],
                r["macd"],
                r.get("bb_width", 0),
                r.get("ema20", price) / price - 1,
                r.get("ema50", price) / price - 1,
                r.get("atr", 0) / price,
                r.get("vol_momentum", 1),
                r.get("adx", 25),
                r.get("stoch", 50),
                r.get("williams_r", -50),
                r.get("cci", 0),
                r.get("vix", 20),
                r.get("spy_ret", 0),
                r.get("qqq_ret", 0),
            ]

            if i + 5 >= len(prices):
                continue
            pct_change = (prices[i + 5] - prices[i]) / prices[i] * 100
            if pct_change > 0.2:
                label = 1
            elif pct_change < -0.2:
                label = 0
            else:
                continue

            X_all.append(features)
            y_all.append(label)

    return np.array(X_all), np.array(y_all)


if __name__ == "__main__":
    print("Fetching data...")
    data = fetch_training_data()

    print("Building features...")
    X, y = build_flat_features(data)
    print(f"Built {len(X)} samples | Class balance: {y.mean():.1%} up")

    # Time-based split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    up_idx = np.where(y_train == 1)[0]
    down_idx = np.where(y_train == 0)[0]
    min_count = min(len(up_idx), len(down_idx))
    idx = np.concatenate([
        np.random.choice(up_idx, min_count, replace=False),
        np.random.choice(down_idx, min_count, replace=False)
    ])
    np.random.shuffle(idx)
    X_train, y_train = X_train[idx], y_train[idx]

    print(f"\nTraining XGBoost on {len(X_train)} samples...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=123
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))

    print("\nResults:")
    print(f"  XGBoost train accuracy: {train_acc:.1%}")
    print(f"  XGBoost val accuracy:   {val_acc:.1%}")
    print("\nFeature importances:")
    feature_names = ["sentiment", "price_change", "rsi", "macd", "bb_width",
                     "ema20_ratio", "ema50_ratio", "atr_ratio", "vol_momentum",
                     "adx", "stoch", "williams_r", "cci", "vix", "spy_ret", "qqq_ret"]
    importances = sorted(zip(feature_names, model.feature_importances_, strict=True), key=lambda x: -x[1])
    for name, imp in importances:
        print(f"  {name}: {imp:.3f}")

    os.makedirs("/app/models", exist_ok=True)
    with open("/app/models/xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("XGBoost model saved → /app/models/xgboost_model.pkl")
