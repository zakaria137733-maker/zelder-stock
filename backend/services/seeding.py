import random
import secrets
from datetime import UTC, datetime, timedelta

from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS

from config import settings
from services.auth import hash_password
from services.influx import get_influx_client
from services.mongo import get_db
from tickers import TICKERS

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

BASE_SCORES = {"AAPL": 72, "TSLA": 41, "NVDA": 88, "MSFT": 68, "GOOGL": 75, "AMZN": 63, "META": 58}
PRICES = {"AAPL": 189.42, "TSLA": 248.17, "NVDA": 876.54, "MSFT": 415.22, "GOOGL": 175.83, "AMZN": 185.60, "META": 490.32}


async def seed_customers() -> list[dict]:
    """Drop and reseed demo customers, returning their generated credentials."""
    db = get_db()
    await db.customers.drop()
    await db.customers.create_index("email", unique=True)

    docs = []
    credentials = []
    for c in CUSTOMERS:
        password = secrets.token_urlsafe(12)
        doc = dict(c)
        doc["password_hash"] = hash_password(password)
        doc["created_at"] = datetime.now(UTC) - timedelta(days=random.randint(30, 365))
        docs.append(doc)
        credentials.append({"email": c["email"], "password": password})

    await db.customers.insert_many(docs)
    return credentials


def seed_influx() -> int:
    """Seed demo sentiment/trades into InfluxDB, returning the point count."""
    client = get_influx_client()
    try:
        buckets_api = client.buckets_api()
        existing = [b.name for b in buckets_api.find_buckets().buckets]
        for bucket in [settings.influx_bucket, settings.influx_trades_bucket]:
            if bucket not in existing:
                buckets_api.create_bucket(bucket_name=bucket, org=settings.influx_org)
                print(f"Created bucket: {bucket}")

        points = []
        now = datetime.now(UTC)

        for ticker in TICKERS:
            base = BASE_SCORES[ticker]
            for h in range(48, 0, -1):
                ts = now - timedelta(hours=h)
                drift = random.gauss(0, 4)
                score = max(10, min(95, base + drift))
                points.append(
                    Point("sentiment")
                    .tag("ticker", ticker)
                    .tag("source", "demo")
                    .field("composite", float(round(score, 2)))
                    .field("score", float(round((score - 50) / 50, 4)))
                    .field("signal_count", float(random.randint(3, 20)))
                    .time(ts)
                )

        for _ in range(200):
            ticker = random.choice(TICKERS)
            side = random.choice(["buy", "sell"])
            price = PRICES[ticker] * (1 + random.uniform(-0.01, 0.01))
            qty = random.randint(5, 150)
            ts = now - timedelta(hours=random.uniform(0, 24))
            points.append(
                Point("trades")
                .tag("ticker", ticker)
                .tag("side", side)
                .tag("customer_id", f"cust_{random.randint(1,8):03d}")
                .tag("is_demo", "true")
                .field("price", round(price, 2))
                .field("quantity", qty)
                .field("total_usd", round(price * qty, 2))
                .time(ts)
            )

        write_api = client.write_api(write_options=SYNCHRONOUS)
        write_api.write(bucket=settings.influx_bucket, record=[p for p in points if p._name == "sentiment"])
        write_api.write(bucket=settings.influx_trades_bucket, record=[p for p in points if p._name == "trades"])
        return len(points)
    finally:
        client.close()
