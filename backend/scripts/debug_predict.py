import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.influx import query_price_history, query_sentiment_history

ticker = "AAPL"
price_history = query_price_history(ticker, hours=168)
sent_history = query_sentiment_history(ticker, hours=168)
sent_map = {s["time"][:13]: s["value"] for s in sent_history}

recent_data = [
    {
        "price": p["price"],
        "time": p["time"],
        "sentiment": sent_map.get(p["time"][:13], 50.0)
    }
    for p in price_history[-20:]
]

print(f"recent_data length: {len(recent_data)}")
print(f"First price: {recent_data[0]['price']}")
print(f"Last price: {recent_data[-1]['price']}")

prices = [r["price"] for r in recent_data]
volumes = [0.0] * len(prices)

print(f"Calling compute_indicators with {len(prices)} points...")
from services.lstm_predictor import compute_indicators
indicators = compute_indicators(prices, volumes)
print(f"RSI length: {len(indicators['rsi'])}")
print("Success!")