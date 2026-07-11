import yfinance as yf
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from config import settings

ticker = "AAPL"
print("Fetching Yahoo Finance data...")
t = yf.Ticker(ticker)
hist = t.history(period="1d", interval="1h")
print(f"Got {len(hist)} rows")

client = InfluxDBClient(url=settings.influx_url,token=settings.influx_token,org=settings.influx_org)
write_api = client.write_api(write_options=SYNCHRONOUS)

points=[]
for ts, row in hist.iterrows():
    p = (
        Point("prices")
        .tag("ticker", ticker)
        .field("close", float(row["Close"]))
        .time(ts.to_pydatetime())
    )
    points.append(p)

print(f"Writing {len(points)} points...")
try:
    write_api.write(bucket="sentiment_scores",record=points)
    print("Write successful!")
except Exception as e:
    print(f"Write FAILED: {e}")

client.close()