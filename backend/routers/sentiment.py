from fastapi import APIRouter, HTTPException
from services.redis_client import cache_get, cache_set
from services.news_collector import fetch_news_for_ticker
from services.sentiment import compute_composite
from services import influx

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]


@router.get("/{ticker}")
async def get_sentiment(ticker: str):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(404, f"Ticker {ticker} not tracked")

    # Try cache first
    cached = cache_get(f"composite:{ticker}")
    signals = await fetch_news_for_ticker(ticker)

    composite = cached["score"] if cached else compute_composite(signals)
    signal_count = len(signals)

    # Build source breakdown
    pos = sum(1 for s in signals if s["label"] == "positive")
    neg = sum(1 for s in signals if s["label"] == "negative")
    neu = sum(1 for s in signals if s["label"] == "neutral")

    label = "bullish" if composite >= 60 else "bearish" if composite <= 40 else "neutral"

    # Get history from InfluxDB
    history = influx.query_sentiment_history(ticker, hours=24)

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
    ticker = ticker.upper()
    return {"ticker": ticker, "history": influx.query_sentiment_history(ticker, hours)}


@router.get("/")
async def get_all_sentiment():
    """Quick overview of all tracked tickers."""
    results = []
    for ticker in TICKERS:
        cached = cache_get(f"composite:{ticker}")
        score = cached["score"] if cached else 50.0
        label = "bullish" if score >= 60 else "bearish" if score <= 40 else "neutral"
        results.append({"ticker": ticker, "composite": score, "label": label})
    return results
