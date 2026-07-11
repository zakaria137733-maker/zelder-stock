import asyncio
import httpx
from config import settings
from services.sentiment import classify_sentiment, compute_composite
from services import influx
from services.redis_client import cache_set

async def force_collect():
    tickers = {
    'AAPL': 'Apple stock',
    'TSLA': 'Tesla stock',
    'NVDA': 'Nvidia stock',
    'MSFT': 'Microsoft stock',
    'GOOGL': 'Google Alphabet stock',
    'AMZN': 'Amazon AWS stock',
    'META': 'Meta Facebook stock',
    }
    for ticker, query in tickers.items():
        params = {'q': query, 'pageSize': 10, 'language': 'en', 'apiKey': settings.news_api_key}
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get('https://newsapi.org/v2/everything', params=params)
            articles = resp.json().get('articles', [])
        signals = []
        for a in articles:
            text = f"{a.get('title','')}. {a.get('description','')}"
            if not text.strip():
                continue
            s = classify_sentiment(text)
            signals.append({**s, 'source': 'newsapi', 'age_hours': 1.0})
        if signals:
            composite = compute_composite(signals)
            influx.write_sentiment(ticker, sum(s['score'] for s in signals)/len(signals), composite, 'newsapi', len(signals))
            cache_set(f'composite:{ticker}', {'score': composite, 'signal_count': len(signals)})
            print(f'{ticker}: {composite:.1f} ({len(signals)} signals)')

asyncio.run(force_collect())