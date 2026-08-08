"""
News collection for tracked tickers — the single home for both the
NewsAPI collector and the Google News RSS collector. Routers and scripts
delegate here instead of forking their own fetch loops.
"""
import asyncio
from datetime import UTC, datetime, timedelta

import feedparser
import httpx

from config import settings
from services import influx
from services.redis_client import add_to_dedup_set, cache_get, cache_set, publish_signal
from services.sentiment import classify_sentiment, compute_composite
from tickers import TICKERS as TRACKED_TICKERS

TICKER_QUERIES = {
    "AAPL": "Apple stock OR iPhone OR Tim Cook",
    "TSLA": "Tesla stock OR Elon Musk electric vehicle",
    "NVDA": "Nvidia stock OR GPU AI chips Jensen Huang",
    "MSFT": "Microsoft stock OR Azure cloud Satya Nadella",
    "GOOGL": "Google stock OR Alphabet Gemini AI",
    "AMZN": "Amazon stock OR AWS cloud Jeff Bezos",
    "META": "Meta stock OR Facebook Instagram Zuckerberg",
}


async def fetch_news_for_ticker(ticker: str) -> list[dict]:
    """Fetch and score news articles for a single ticker."""
    loop = asyncio.get_event_loop()
    cache_key = f"news:{ticker}"
    cached = await loop.run_in_executor(None, cache_get, cache_key)
    if cached:
        return cached

    if not settings.news_api_key:
        print("NEWS_API_KEY not set — returning empty results")
        return []

    query = TICKER_QUERIES.get(ticker, ticker)
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 20,
        "from": (datetime.now(UTC) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S"),
        "apiKey": settings.news_api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        print(f"NewsAPI error for {ticker}: {e}")
        return []

    articles = data.get("articles", [])
    signals = []

    for article in articles:
        url_str = article.get("url", "")
        if not await loop.run_in_executor(None, add_to_dedup_set, url_str):
            continue  # already processed

        title = article.get("title") or ""
        description = article.get("description") or ""
        text = f"{title}. {description}".strip()

        if not text or text == ".":
            continue

        published = article.get("publishedAt", "")
        try:
            pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            age_hours = (datetime.now(UTC) - pub_dt).total_seconds() / 3600
        except Exception:
            age_hours = 1.0

        sentiment = classify_sentiment(text)

        signal = {
            "ticker": ticker,
            "source": "newsapi",
            "title": title,
            "url": url_str,
            "published_at": published,
            "age_hours": round(age_hours, 2),
            "score": sentiment["score"],
            "label": sentiment["label"],
            "confidence": sentiment["confidence"],
            "source_name": article.get("source", {}).get("name", "Unknown"),
        }
        signals.append(signal)

        # Publish to Redis pub/sub for WebSocket live feed
        await loop.run_in_executor(None, publish_signal, signal)

    await loop.run_in_executor(None, cache_set, cache_key, signals, 1800)  # cache 30 min
    return signals


async def collect_and_score_all() -> dict[str, float]:
    """
    Main collection job — runs all tickers, writes composites to InfluxDB.
    Returns {ticker: composite_score}.
    """
    results = {}
    loop = asyncio.get_event_loop()

    for ticker in TRACKED_TICKERS:
        print(f"Collecting {ticker}...")
        signals = await fetch_news_for_ticker(ticker)

        if signals:
            composite = compute_composite(signals)
            results[ticker] = composite

            # Write composite to InfluxDB
            influx.write_sentiment(
                ticker=ticker,
                score=sum(s["score"] for s in signals) / len(signals),
                composite=composite,
                source="newsapi",
                signal_count=len(signals),
            )

            # Cache composite score with 30-min TTL
            await loop.run_in_executor(
                None, cache_set, f"composite:{ticker}",
                {"score": composite, "signal_count": len(signals)}, 1800,
            )
            print(f"  {ticker}: {composite:.1f} ({len(signals)} signals)")
        else:
            results[ticker] = 50.0  # neutral fallback

    return results


def fetch_google_news(ticker: str, max_entries: int = 12) -> list[dict]:
    """Fetch and score Google News RSS for one ticker. Blocking (feedparser)."""
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    signals = []
    for entry in feed.entries[:max_entries]:
        text = (entry.get("title", "") + ". " + entry.get("summary", "")).strip()[:500]
        if not text:
            continue
        s = classify_sentiment(text)
        source_name = "Google News"
        src = entry.get("source", {})
        if hasattr(src, "get"):
            source_name = src.get("title", "Google News")
        signals.append({
            "ticker": ticker,
            "source": "google_news",
            "source_name": source_name,
            "title": entry.get("title", ""),
            "url": entry.get("link", ""),
            "published_at": entry.get("published", ""),
            "age_hours": 1.0,
            "score": s["score"],
            "label": s["label"],
            "confidence": s["confidence"],
        })
    return signals


def collect_google_sentiment(ticker: str, max_entries: int = 12) -> dict:
    """Fetch, score, persist, and cache one ticker's Google News sentiment.

    The single shared implementation behind /api/admin/collect, the Temporal
    free-collector, and scripts/free_collect.py. Returns the composite payload.
    """
    signals = fetch_google_news(ticker, max_entries=max_entries)
    composite = 50.0
    if signals:
        composite = compute_composite(signals)
        avg_score = sum(s["score"] for s in signals) / len(signals)
        influx.write_sentiment(ticker, avg_score, composite, "google_news", len(signals))
        cache_set(f"signals:{ticker}", signals, ttl=3600)
        cache_set(f"composite:{ticker}", {"score": composite, "signal_count": len(signals)}, ttl=3600)
    return {
        "ticker": ticker,
        "composite": composite,
        "signal_count": len(signals),
        "signals": signals,
    }
