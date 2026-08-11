import asyncio

from fastapi import APIRouter

from services import influx
from services.news_collector import fetch_news_for_ticker
from services.redis_client import cache_get
from services.sentiment import compute_composite
from tickers import TICKERS, validate_ticker

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])


@router.get("/{ticker}")
async def get_sentiment(ticker: str):
    ticker = validate_ticker(ticker)
    loop = asyncio.get_running_loop()

    # Try cache first
    cached = await loop.run_in_executor(None, cache_get, f"composite:{ticker}")
    signals = await fetch_news_for_ticker(ticker)

    composite = cached["score"] if cached else compute_composite(signals)
    signal_count = len(signals)

    # Build source breakdown
    pos = sum(1 for s in signals if s["label"] == "positive")
    neg = sum(1 for s in signals if s["label"] == "negative")
    neu = sum(1 for s in signals if s["label"] == "neutral")

    label = "bullish" if composite >= 60 else "bearish" if composite <= 40 else "neutral"

    # Get history from InfluxDB
    history = await loop.run_in_executor(None, influx.query_sentiment_history, ticker, 24)

    return {
        "ticker": ticker,
        "composite": composite,
        "label": label,
        "signal_count": signal_count,
        "breakdown": {"positive": pos, "negative": neg, "neutral": neu},
        "history": history,
    }


@router.get("/{ticker}/history")
async def get_history(ticker: str, hours: int = 24):
    ticker = validate_ticker(ticker)
    loop = asyncio.get_running_loop()
    history = await loop.run_in_executor(None, influx.query_sentiment_history, ticker, hours)
    return {"ticker": ticker, "history": history}


@router.get("/")
async def get_all_sentiment():
    """Quick overview of all tracked tickers."""
    loop = asyncio.get_running_loop()
    results = []
    for ticker in TICKERS:
        cached = await loop.run_in_executor(None, cache_get, f"composite:{ticker}")
        score = cached["score"] if cached else 50.0
        label = "bullish" if score >= 60 else "bearish" if score <= 40 else "neutral"
        results.append({"ticker": ticker, "composite": score, "label": label})
    return results
