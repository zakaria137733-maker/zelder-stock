import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import random
from datetime import datetime, timezone, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from config import settings

CUSTOMERS = [
    {"name": "Sarah Chen", "email": "sarah.chen@example.com", "portfolio_value": 142800, "watchlist": ["AAPL", "NVDA"], "risk_profile": "conservative", "sentiment_score": 87},
    {"name": "Marcus Webb", "email": "marcus.webb@example.com", "portfolio_value": 58200, "watchlist": ["TSLA", "AAPL"], "risk_profile": "aggressive", "sentiment_score": 72},
    {"name": "Priya Nair", "email": "priya.nair@example.com", "portfolio_value": 214500, "watchlist": ["NVDA", "MSFT", "GOOGL"], "risk_profile": "moderate", "sentiment_score": 91},
    {"name": "James O'Brien", "email": "james.obrien@example.com", "portfolio_value": 31000, "watchlist": ["TSLA"], "risk_profile": "aggressive", "sentiment_score": 38},
    {"name": "Yuki Tanaka", "email": "yuki.tanaka@example.com", "portfolio_value": 98700, "watchlist": ["AAPL", "AMZN", "META"], "risk_profile": "moderate", "sentiment_score": 65},
    {"name": "Alex Rivera", "email": "alex.rivera@example.com", "portfolio_value": 175300, "watchlist": ["NVDA", "MSFT"], "risk_profile": "conservative", "sentiment_score": 79},
    {"name": "Dana Kim", "email": "dana.kim@example.com", "portfolio_value": 22400, "watchlist": ["TSLA", "AMZN"], "risk_profile": "aggressive", "sentiment_score": 44},
    {"name": "Omar Hassan", "email": "omar.hassan@example.com", "portfolio_value": 310000, "watchlist": ["AAPL", "MSFT", "GOOGL", "NVDA"], "risk_profile": "conservative", "sentiment_score": 83},
]

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]
BASE_SCORES = {"AAPL": 72, "TSLA": 41, "NVDA": 88, "MSFT": 68, "GOOGL": 75, "AMZN": 63, "META": 58}
PRICES = {"AAPL": 189.42, "TSLA": 248.17, "NVDA": 876.54, "MSFT": 415.22, "GOOGL": 175.83, "AMZN": 185.60, "META": 490.32}


async def seed_mongo():
    client = AsyncIOMotorClient(settings.mongo_uri)
    db = client.sentimentiq

    await db.customers.drop()
    await db.customers.create_index("email", unique=True)

    docs = [{**c, "created_at": datetime.now(timezone.utc) - timedelta(days=random.randint(30, 365))} for c in CUSTOMERS]
    await db.customers.insert_many(docs)
    print(f"Seeded {len(docs)} customers into MongoDB")
    client.close()


def seed_influx():
    client = InfluxDBClient(url=settings.influx_url, token=settings.influx_token, org=settings.influx_org)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    buckets_api = client.buckets_api()
    existing = [b.name for b in buckets_api.find_buckets().buckets]
    for bucket in ["sentiment_scores", "stock_trades"]:
        if bucket not in existing:
            from influxdb_client.domain.bucket import Bucket
            buckets_api.create_bucket(bucket_name=bucket, org=settings.influx_org)
            print(f"Created bucket: {bucket}")

    points = []
    now = datetime.now(timezone.utc)

    for ticker in TICKERS:
        base = BASE_SCORES[ticker]
        for h in range(48, 0, -1):
            ts = now - timedelta(hours=h)
            drift = random.gauss(0, 4)
            score = max(10, min(95, base + drift))
            p = (
                Point("sentiment")
                .tag("ticker", ticker)
                .tag("source", "newsapi")
                .field("composite", float(round(score, 2)))
                .field("score", float(round((score - 50) / 50, 4)))
                .field("signal_count", float(random.randint(3, 20)))
                .time(ts)
            )
            points.append(p)

    for _ in range(200):
        ticker = random.choice(TICKERS)
        side = random.choice(["buy", "sell"])
        price = PRICES[ticker] * (1 + random.uniform(-0.01, 0.01))
        qty = random.randint(5, 150)
        ts = now - timedelta(hours=random.uniform(0, 24))
        p = (
            Point("trades")
            .tag("ticker", ticker)
            .tag("side", side)
            .tag("customer_id", f"cust_{random.randint(1,8):03d}")
            .field("price", round(price, 2))
            .field("quantity", qty)
            .field("total_usd", round(price * qty, 2))
            .time(ts)
        )
        points.append(p)

    write_api.write(bucket="sentiment_scores", record=[p for p in points if p._name == "sentiment"])
    write_api.write(bucket="stock_trades", record=[p for p in points if p._name == "trades"])
    print(f"Seeded {len(points)} points into InfluxDB")
    client.close()


async def main():
    print("Seeding databases...")
    await seed_mongo()
    seed_influx()
    print("Its Done.")


if __name__ == "__main__":
    asyncio.run(main())
