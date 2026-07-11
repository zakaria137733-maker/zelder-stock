import asyncio
import json
import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from config import settings
from services.sentiment import classify_sentiment
from services.redis_client import cache_set, cache_get
import redis.asyncio as aioredis

router = APIRouter(tags=["signals"])

TICKERS = ["AAPL","TSLA","NVDA","MSFT","GOOGL","AMZN","META"]
TICKER_QUERIES = {
    "AAPL":"Apple stock","TSLA":"Tesla stock","NVDA":"Nvidia stock",
    "MSFT":"Microsoft stock","GOOGL":"Google Alphabet stock",
    "AMZN":"Amazon stock","META":"Meta Facebook stock",
}

async def fetch_signals_for_ticker(ticker:str)->list[dict]:
    cached=cache_get(f"signals:{ticker}")
    if cached:
        return cached
    if not settings.news_api_key:
        return []
    params = {"q": TICKER_QUERIES.get(ticker,ticker),"pageSize":10,"language":"en","apiKey":settings.news_api_key}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp=await client.get("https://newsapi.org/v2/everything", params=params)
            articles=resp.json().get("articles",[])
    except Exception as e:
        print(f"NewsAPI error:{e}")
        return []
    signals = []
    for a in articles:
        text = f"{a.get('title','')}.{a.get('description','')}".strip()
        if not text or text==".":
            continue
        s = classify_sentiment(text)
        signals.append({
            "ticker": ticker,"source":"newsapi",
            "source_name":a.get("source",{}).get("name","NewsAPI"),
            "title":a.get("title",""),"url":a.get("url",""),
            "published_at":a.get("publishedAt",""),"age_hours":1.0,
            "score":s["score"],"label":s["label"],"confidence":s["confidence"],
        })
    if signals:
        cache_set(f"signals:{ticker}",signals,ttl=1800)
    return signals

@router.get("/api/signals")
async def get_signals(ticker: str | None=None, limit: int=30):
    tickers = [ticker.upper()] if ticker else TICKERS[:4]
    all_signals=[]
    for t in tickers:
        all_signals.extend(await fetch_signals_for_ticker(t))
    return sorted(all_signals,key=lambda s:s.get("age_hours",99))[:limit]

@router.websocket("/ws/signals")
async def signals_websocket(ws: WebSocket):
    await ws.accept()
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    pubsub = r.pubsub()
    await pubsub.subscribe("signals")
    try:
        while True:
            message = await asyncio.wait_for(pubsub.get_message(ignore_subscribe_messages=True), timeout=30)
            if message and message["type"]=="message":
                await ws.send_text(message["data"])
            else:
                await ws.send_text(json.dumps({"type": "heartbeat"}))
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        await pubsub.unsubscribe("signals")
        await r.aclose()
