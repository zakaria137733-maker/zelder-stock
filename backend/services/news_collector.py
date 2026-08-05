"""
NewsAPI collector — fetches headlines for each tracked ticker,
runs sentiment analysis, writes to InfluxDB, and caches in Redis.
"""
from datetime import UTC, datetime, timedelta

import httpx

from config import settings
from services import influx
from services.redis_client import add_to_dedup_set, cache_get, cache_set, publish_signal
from services.sentiment import classify_sentiment, compute_composite

TRACKED_TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]

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
    cache_key = f"news:{ticker}"
    cached = cache_get(cache_key)
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
        if not add_to_dedup_set(url_str):
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
        publish_signal(signal)

    cache_set(cache_key, signals, ttl=1800)  # cache 30 min
    return signals


async def collect_and_score_all() -> dict[str, float]:
    """
    Main collection job — runs all tickers, writes composites to InfluxDB.
    Returns {ticker: composite_score}.
    """
    results = {}

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
            cache_set(f"composite:{ticker}", {"score": composite, "signal_count": len(signals)}, ttl=1800)
            print(f"  {ticker}: {composite:.1f} ({len(signals)} signals)")
        else:
            results[ticker] = 50.0  # neutral fallback

    return results
