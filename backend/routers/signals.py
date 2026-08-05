import asyncio
import json

import feedparser
import redis.asyncio as aioredis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import settings
from services.redis_client import cache_get, cache_set
from services.sentiment import classify_sentiment
from tickers import TICKERS, validate_ticker

router = APIRouter(tags=["signals"])

TICKER_QUERIES = {
    "AAPL":"Apple stock","TSLA":"Tesla stock","NVDA":"Nvidia stock",
    "MSFT":"Microsoft stock","GOOGL":"Google Alphabet stock",
    "AMZN":"Amazon stock","META":"Meta Facebook stock",
}


def _fetch_google_news(ticker: str) -> list[dict]:
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    signals = []
    for entry in feed.entries[:12]:
        text = entry.get("title", "") + ". " + entry.get("summary", "")
        text = text[:500]
        if not text.strip():
            continue
        s = classify_sentiment(text)
        source_name = "Google News"
        try:
            if hasattr(entry.get("source", {}), "get"):
                source_name = entry["source"].get("title", "Google News")
        except Exception:
            pass
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


async def fetch_signals_for_ticker(ticker: str) -> list[dict]:
    cached = cache_get(f"signals:{ticker}")
    if cached:
        return cached

    try:
        signals = await asyncio.get_event_loop().run_in_executor(None, _fetch_google_news, ticker)
        if signals:
            cache_set(f"signals:{ticker}", signals, ttl=3600)
        return signals
    except Exception as e:
        print(f"Google News error for {ticker}: {e}")
        return []


@router.get("/api/signals")
async def get_signals(ticker: str | None = None, limit: int = 30):
    tickers = [validate_ticker(ticker)] if ticker else TICKERS
    all_signals = []
    for t in tickers:
        all_signals.extend(await fetch_signals_for_ticker(t))
    return sorted(all_signals, key=lambda s: s.get("age_hours", 99))[:limit]


@router.websocket("/ws/signals")
async def signals_websocket(ws: WebSocket):
    await ws.accept()
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    pubsub = r.pubsub()
    await pubsub.subscribe("signals")
    try:
        while True:
            message = await asyncio.wait_for(pubsub.get_message(ignore_subscribe_messages=True), timeout=30)
            if message and message["type"] == "message":
                await ws.send_text(message["data"])
            else:
                await ws.send_text(json.dumps({"type": "heartbeat"}))
            await asyncio.sleep(0.1)
    except (TimeoutError, WebSocketDisconnect):
        pass
    finally:
        await pubsub.unsubscribe("signals")
        await r.aclose()
