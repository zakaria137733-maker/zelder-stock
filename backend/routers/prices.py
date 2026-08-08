import asyncio

from fastapi import APIRouter

from services import influx
from services.redis_client import cache_get
from tickers import TICKERS, validate_ticker

router = APIRouter(prefix="/api/prices", tags=["prices"])


@router.get("/")
async def get_all_prices():
    loop = asyncio.get_event_loop()
    results = []
    for ticker in TICKERS:
        cached = await loop.run_in_executor(None, cache_get, f"price:{ticker}")
        results.append(cached or {"ticker": ticker, "price": None, "change_pct": 0})
    return results


@router.get("/{ticker}/history")
async def get_price_history(ticker: str, hours: int = 24):
    ticker = validate_ticker(ticker)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, influx.query_price_history, ticker, hours)


@router.get("/{ticker}")
async def get_price(ticker: str):
    ticker = validate_ticker(ticker)
    loop = asyncio.get_event_loop()
    cached = await loop.run_in_executor(None, cache_get, f"price:{ticker}")
    if cached:
        return cached
    return {"ticker": ticker, "price": None, "change_pct": 0}
