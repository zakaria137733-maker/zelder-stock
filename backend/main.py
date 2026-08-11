import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import (
    admin,
    alerts,
    auth,
    customers,
    predictions,
    prices,
    sentiment,
    signals,
    transactions,
)
from services import influx, mongo
from tickers import TICKERS

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        db = mongo.get_db()
        await db.customers.create_index("email", unique=True)
        await db.customers.create_index("watchlist")
        await db.customers.create_index([("sentiment_score", -1)])
        logger.info("ZelderStock API ready (MongoDB connected)")
    except Exception as e:
        logger.warning("MongoDB unavailable at startup, continuing: %s", e)
    influx.ensure_buckets()
    yield
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
app.include_router(admin.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ZelderStock", "last_collect": admin.get_collect_status()}


@app.get("/api/tickers")
async def list_tickers():
    return TICKERS
