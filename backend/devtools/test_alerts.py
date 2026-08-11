import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services import influx
from services.alerts import WINDOW_HOURS

tickers = ["AMZN", "META"]

for ticker in tickers:
    history = influx.query_sentiment_history(ticker, hours=WINDOW_HOURS + 1)
    print(f"\n{ticker}: {len(history)} points")
    for h in history:
        print(f"  {h['time']} → {h['value']}")
