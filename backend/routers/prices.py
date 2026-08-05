from fastapi import APIRouter

from services import influx
from services.redis_client import cache_get
from tickers import TICKERS, validate_ticker

router = APIRouter(prefix="/api/prices", tags=["prices"])


@router.get("/")
async def get_all_prices():
    results = []
    for ticker in TICKERS:
        cached = cache_get(f"price:{ticker}") or {"ticker": ticker, "price": None, "change_pct": 0}
        results.append(cached)
    return results


@router.get("/{ticker}/history")
async def get_price_history(ticker: str, hours: int = 24):
    ticker = validate_ticker(ticker)
    return influx.query_price_history(ticker, hours)


@router.get("/{ticker}")
async def get_price(ticker: str):
    ticker = validate_ticker(ticker)
    cached = cache_get(f"price:{ticker}")
    if cached:
        return cached
    return {"ticker": ticker, "price": None, "change_pct": 0}
