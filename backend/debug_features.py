from services.lstm_predictor import fetch_training_data, build_sequences
import numpy as np

data = fetch_training_data()
X, y = build_sequences(data)

print(f"X shape: {X.shape}")
print(f"\nFeature statistics (before normalization):")
feature_names = ["sentiment", "price", "price_change", "volume", "rsi", "macd",
                 "bb_upper", "bb_lower", "bb_width", "ma20", "ema20", "ema50",
                 "atr", "vol_momentum", "adx", "obv", "stoch", "williams_r",
                 "cci", "vwap", "spy_ret", "qqq_ret", "vix"]

for i in range(X.shape[2]):
    vals = X[:, :, i].flatten()
    name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
    print(f"  {name:15s}: min={vals.min():12.2f} max={vals.max():12.2f} mean={vals.mean():10.2f} std={vals.std():10.2f}")