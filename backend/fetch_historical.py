import yfinance as yf
from datetime import datetime, timezone
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from config import settings

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]

def fetch_and_store():
    client = InfluxDBClient(url=settings.influx_url, token=settings.influx_token, org=settings.influx_org)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    total = 0
    for ticker in TICKERS:
        print(f"Fetching {ticker}...")
        t = yf.Ticker(ticker)
        # Max period for hourly data is 730 days but free tier gives 60 days
        hist = t.history(period="60d", interval="1h")
        points = []
        for ts, row in hist.iterrows():
            p = (
                Point("prices")
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

    client.close()
    print(f"\nTotal: {total} points written to InfluxDB")

if __name__ == "__main__":
    fetch_and_store()