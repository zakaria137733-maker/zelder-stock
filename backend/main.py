import asyncio
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import alerts, auth, customers, predictions, prices, sentiment, signals, transactions
from services import mongo
from services.auth import require_admin
from tickers import TICKERS

_last_collect = {"time": None, "status": "idle"}


def _run_lightweight_collect():
    try:
        _last_collect["status"] = "running"
        from services.news_collector import collect_google_sentiment

        for ticker in TICKERS:
            try:
                result = collect_google_sentiment(ticker)
                print(f"  {ticker}: {result['composite']:.1f} ({result['signal_count']} signals)")
            except Exception as e:
                print(f"  {ticker} error: {e}")

        _last_collect["time"] = datetime.now(UTC).isoformat()
        _last_collect["status"] = "idle"
        print("Collection completed")
    except Exception as e:
        _last_collect["status"] = f"error: {e}"
        print(f"Collection error: {e}")
        import traceback
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        db = mongo.get_db()
        await db.customers.create_index("email", unique=True)
        await db.customers.create_index("watchlist")
        await db.customers.create_index([("sentiment_score", -1)])
        print("SentimentIQ API ready (MongoDB connected)")
    except Exception as e:
        print(f"WARNING: MongoDB unavailable at startup, continuing: {e}")
    yield
    await mongo.close()


app = FastAPI(
    title="SentimentIQ",
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
app.include_router(auth.admin_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "SentimentIQ", "last_collect": _last_collect}


@app.get("/api/tickers")
async def list_tickers():
    return TICKERS


@app.post("/api/admin/collect")
async def trigger_collect(_admin=Depends(require_admin)):
    asyncio.get_event_loop().run_in_executor(None, _run_lightweight_collect)
    return {"ok": True, "message": "Collection started"}


@app.get("/api/admin/collect/status")
async def collect_status(_admin=Depends(require_admin)):
    return _last_collect


@app.post("/api/admin/seed")
async def seed_all(_admin=Depends(require_admin)):
    from services import seeding

    errors = []
    customer_count = 0
    influx_count = 0
    demo_credentials = []

    try:
        demo_credentials = await seeding.seed_customers()
        customer_count = len(demo_credentials)
    except Exception as e:
        errors.append(f"MongoDB: {str(e)}")

    try:
        influx_count = seeding.seed_influx()
    except Exception as e:
        errors.append(f"InfluxDB: {str(e)}")

    return {
        "ok": len(errors) == 0,
        "customers": customer_count,
        "influx_points": influx_count,
        "demo_credentials": demo_credentials,
        "errors": errors,
    }
