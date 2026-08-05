import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timezone, timedelta
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from config import settings

TRAJECTORIES = {
    "AAPL": [68, 70, 71, 69, 68, 65, 60, 55, 48, 44, 42, 43],
    "TSLA": [42, 43, 41, 40, 44, 48, 55, 60, 62, 58, 54, 54],
    "NVDA": [85, 84, 82, 80, 75, 68, 60, 55, 50, 44, 43, 43],
    "MSFT": [65, 66, 65, 67, 66, 65, 64, 63, 62, 46, 45, 46],
    "GOOGL": [50, 51, 50, 49, 48, 47, 45, 42, 41, 40, 40, 40],
    "AMZN": [67, 66, 65, 64, 62, 58, 52, 48, 44, 40, 39, 38],
    "META": [66, 65, 64, 62, 60, 58, 55, 52, 48, 44, 42, 41],
}

def seed():
    client = InfluxDBClient(url=settings.influx_url, token=settings.influx_token, org=settings.influx_org)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    now = datetime.now(timezone.utc)
    points = []

    for ticker, scores in TRAJECTORIES.items():
        n = len(scores)
        for i, score in enumerate(scores):
            # Spread points evenly across last 3 hours
            minutes_ago = int((n - i) * (180 / n))
            ts = now - timedelta(minutes=minutes_ago)
            p = (
                Point("sentiment")
                .tag("ticker", ticker)
                .tag("source", "finbert")
                .field("composite", float(score))
                .field("score", float((score - 50) / 50))
                .field("signal_count", float(15))
                .time(ts)
            )
            points.append(p)

    write_api.write(bucket="sentiment_scores", record=points)
    print(f"Seeded {len(points)} points within 3h window")
    client.close()

if __name__ == "__main__":
    seed()