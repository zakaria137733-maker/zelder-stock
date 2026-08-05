import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yfinance as yf
from datetime import datetime, timezone
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from config import settings

TICKERS = [
    "AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META",
    "AMD", "INTC", "QCOM", "AVGO", "MU", "SMCI", "ARM",
    "JPM", "BAC", "GS", "MS", "V", "MA", "BRK-B",
    "JNJ", "PFE", "UNH", "MRNA", "ABBV", "LLY",
    "XOM", "CVX", "SLB", "COP",
    "WMT", "HD", "NKE", "MCD", "SBUX", "COST",
    "CRM", "SNOW", "PLTR", "NET", "DDOG", "NOW",
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE",
]
MARKET_TICKERS = ["^VIX"]

def fetch_and_store():
    client = InfluxDBClient(url=settings.influx_url, token=settings.influx_token, org=settings.influx_org)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    total = 0
    for ticker in TICKERS:
        print(f"Fetching {ticker}...")
        t = yf.Ticker(ticker)
        hist = t.history(period="2y", interval="1d")
        points = []
        for ts, row in hist.iterrows():
            p = (
                Point("prices_daily")
                .tag("ticker", ticker)
                .field("close", float(row["Close"]))
                .field("volume", float(row["Volume"]))
                .field("high", float(row["High"]))
                .field("low", float(row["Low"]))
                .time(ts.to_pydatetime())
            )
            points.append(p)
        if points:
            write_api.write(bucket="sentiment_scores", record=points)
            print(f"  {ticker}: {len(points)} hourly price points written")
            total += len(points)
    
    print("\nFetching market indices...")
    for ticker in MARKET_TICKERS:
        safe_name = ticker.replace("^", "")
        print(f"Fetching {ticker}...")
        t = yf.Ticker(ticker)
        hist = t.history(period="2y", interval="1d")
        points = []
        for ts, row in hist.iterrows():
            p = (
                Point("market_index")
                .tag("ticker", safe_name)
                .field("close", float(row["Close"]))
                .field("volume", float(row.get("Volume", 0)))
                .time(ts.to_pydatetime())
            )
            points.append(p)
        if points:
            write_api.write(bucket="sentiment_scores", record=points)
            print(f"  {safe_name}: {len(points)} points written")
            total += len(points)        

    client.close()
    print(f"\nTotal: {total} points written to InfluxDB")

if __name__ == "__main__":
    fetch_and_store()