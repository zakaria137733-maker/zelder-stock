from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import sentiment, customers, transactions, signals, prices, alerts, predictions
from services import mongo
from routers import auth
import os, asyncio
from datetime import datetime, timezone, timedelta
import random
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from config import settings
from services.auth import hash_password

_collect_task = None
_last_collect = {"time": None, "results": []}


def _run_collection():
    try:
        from free_collect import collect_all
        collect_all()
        _last_collect["time"] = datetime.now(timezone.utc).isoformat()
        _last_collect["results"] = ["done"]
        print("Collection completed successfully")
    except Exception as e:
        print(f"Collection error: {e}")
        import traceback
        traceback.print_exc()


async def _periodic_collection():
    await asyncio.sleep(10)
    asyncio.get_event_loop().run_in_executor(None, _run_collection)
    while True:
        await asyncio.sleep(300)
        asyncio.get_event_loop().run_in_executor(None, _run_collection)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = mongo.get_db()
    await db.customers.create_index("email", unique=True)
    await db.customers.create_index("watchlist")
    await db.customers.create_index([("sentiment_score", -1)])
    print("ZelderStock API ready")

    task = asyncio.create_task(_periodic_collection())
    yield
    task.cancel()
    await mongo.close()


app = FastAPI(
    title="ZelderStock",
    description="Stock market sentiment analysis platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://zelder-stock-f.onrender.com",
        os.getenv("FRONTEND_URL", ""),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sentiment.router)
app.include_router(customers.router)
app.include_router(transactions.router)
app.include_router(signals.router)
app.include_router(prices.router)
app.include_router(alerts.router)
app.include_router(predictions.router)
app.include_router(auth.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ZelderStock", "last_collect": _last_collect["time"]}


@app.get("/api/tickers")
async def list_tickers():
    return ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]


@app.post("/api/admin/collect")
async def trigger_collect():
    asyncio.get_event_loop().run_in_executor(None, _run_collection)
    return {"ok": True, "message": "Collection started in background"}


@app.get("/api/admin/collect/status")
async def collect_status():
    return {"last_collect": _last_collect["time"], "results": _last_collect["results"]}


@app.post("/api/admin/seed")
async def seed_all():
    errors = []
    customer_count = 0
    influx_count = 0

    try:
        db = mongo.get_db()

        CUSTOMERS = [
            {"name": "Sarah Chen", "email": "sarah.chen@example.com", "portfolio_value": 142800, "watchlist": ["AAPL", "NVDA"], "risk_profile": "conservative", "sentiment_score": 87, "password": "demo123"},
            {"name": "Marcus Webb", "email": "marcus.webb@example.com", "portfolio_value": 58200, "watchlist": ["TSLA", "AAPL"], "risk_profile": "aggressive", "sentiment_score": 72, "password": "demo123"},
            {"name": "Priya Nair", "email": "priya.nair@example.com", "portfolio_value": 214500, "watchlist": ["NVDA", "MSFT", "GOOGL"], "risk_profile": "moderate", "sentiment_score": 91, "password": "demo123"},
            {"name": "James O'Brien", "email": "james.obrien@example.com", "portfolio_value": 31000, "watchlist": ["TSLA"], "risk_profile": "aggressive", "sentiment_score": 38, "password": "demo123"},
            {"name": "Yuki Tanaka", "email": "yuki.tanaka@example.com", "portfolio_value": 98700, "watchlist": ["AAPL", "AMZN", "META"], "risk_profile": "moderate", "sentiment_score": 65, "password": "demo123"},
            {"name": "Alex Rivera", "email": "alex.rivera@example.com", "portfolio_value": 175300, "watchlist": ["NVDA", "MSFT"], "risk_profile": "conservative", "sentiment_score": 79, "password": "demo123"},
            {"name": "Dana Kim", "email": "dana.kim@example.com", "portfolio_value": 22400, "watchlist": ["TSLA", "AMZN"], "risk_profile": "aggressive", "sentiment_score": 44, "password": "demo123"},
            {"name": "Omar Hassan", "email": "omar.hassan@example.com", "portfolio_value": 310000, "watchlist": ["AAPL", "MSFT", "GOOGL", "NVDA"], "risk_profile": "conservative", "sentiment_score": 83, "password": "demo123"},
        ]

        await db.customers.drop()
        await db.customers.create_index("email", unique=True)
        docs = []
        for c in CUSTOMERS:
            doc = {k: v for k, v in c.items() if k != "password"}
            doc["password_hash"] = hash_password(c["password"])
            doc["created_at"] = datetime.now(timezone.utc) - timedelta(days=random.randint(30, 365))
            docs.append(doc)
        await db.customers.insert_many(docs)
        customer_count = len(docs)
    except Exception as e:
        errors.append(f"MongoDB: {str(e)}")

    try:
        TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]
        BASE_SCORES = {"AAPL": 72, "TSLA": 41, "NVDA": 88, "MSFT": 68, "GOOGL": 75, "AMZN": 63, "META": 58}
        PRICES = {"AAPL": 189.42, "TSLA": 248.17, "NVDA": 876.54, "MSFT": 415.22, "GOOGL": 175.83, "AMZN": 185.60, "META": 490.32}

        influx_client = InfluxDBClient(url=settings.influx_url, token=settings.influx_token, org=settings.influx_org)
        buckets_api = influx_client.buckets_api()
        existing = [b.name for b in buckets_api.find_buckets().buckets]
        for bucket_name in ["sentiment_scores", "stock_trades"]:
            if bucket_name not in existing:
                buckets_api.create_bucket(bucket_name=bucket_name, org=settings.influx_org)
        write_api = influx_client.write_api(write_options=SYNCHRONOUS)

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
        influx_client.close()
        influx_count = len(points)
    except Exception as e:
        errors.append(f"InfluxDB: {str(e)}")

    return {"ok": len(errors) == 0, "customers": customer_count, "influx_points": influx_count, "errors": errors}
