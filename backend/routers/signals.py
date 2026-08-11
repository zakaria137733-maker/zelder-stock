import asyncio
import json

import redis.asyncio as aioredis
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from config import settings
from services.news_collector import fetch_google_news
from services.redis_client import cache_get, cache_set
from tickers import TICKERS, validate_ticker

router = APIRouter(tags=["signals"])

# Back-compat alias: main.py historically imported this symbol.
_fetch_google_news = fetch_google_news


async def fetch_signals_for_ticker(ticker: str) -> list[dict]:
    loop = asyncio.get_running_loop()
    cached = await loop.run_in_executor(None, cache_get, f"signals:{ticker}")
    if cached:
        return cached

    try:
        signals = await loop.run_in_executor(None, fetch_google_news, ticker)
        if signals:
            await loop.run_in_executor(None, cache_set, f"signals:{ticker}", signals, 3600)
        return signals
    except Exception as e:
        print(f"Google News error for {ticker}: {e}")
        return []


@router.get("/api/signals")
async def get_signals(ticker: str | None = None, limit: int = Query(30, ge=1, le=200)):
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
