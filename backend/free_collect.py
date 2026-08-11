import asyncio
import feedparser
import yfinance as yf
from datetime import datetime, timezone
from services.sentiment import classify_sentiment, compute_composite
from services import influx
from services.redis_client import cache_set

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]


def fetch_google_news(ticker: str) -> list[dict]:
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    signals = []
    for entry in feed.entries[:15]:
        text = entry.get("title", "") + ". " + entry.get("summary", "")
        text = text[:500]
        if not text.strip():
            continue
        s = classify_sentiment(text)
        signals.append({
            "ticker": ticker,
            "source": "google_news",
            "source_name": entry.get("source", {}).get("title", "Google News") if hasattr(entry.get("source", {}), "get") else "Google News",
            "title": entry.get("title", ""),
            "url": entry.get("link", ""),
            "published_at": entry.get("published", ""),
            "age_hours": 1.0,
            "score": s["score"],
            "label": s["label"],
            "confidence": s["confidence"],
        })
    return signals


def fetch_price_data(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d", interval="1h")
        info = t.fast_info
        current_price = float(info.last_price) if hasattr(info, "last_price") else None
        prev_close = float(info.previous_close) if hasattr(info, "previous_close") else None
        change_pct = ((current_price - prev_close) / prev_close * 100) if current_price and prev_close else 0

        from influxdb_client import InfluxDBClient, Point
        from influxdb_client.client.write_api import SYNCHRONOUS
        from config import settings

        client = InfluxDBClient(url=settings.influx_url, token=settings.influx_token, org=settings.influx_org)
        write_api = client.write_api(write_options=SYNCHRONOUS)

        points = []
        for ts, row in hist.iterrows():
            p = (
                Point("prices")
                .tag("ticker", ticker)
                .field("open", float(row["Open"]))
                .field("high", float(row["High"]))
                .field("low", float(row["Low"]))
                .field("close", float(row["Close"]))
                .field("volume", float(row["Volume"]))
                .time(ts.to_pydatetime())
            )
            points.append(p)

        if points:
            write_api.write(bucket="sentiment_scores", record=points)
            print(f"  Wrote {len(points)} price points to InfluxDB")

        client.close()

        return {
            "ticker": ticker,
            "price": round(current_price, 2) if current_price else None,
            "prev_close": round(prev_close, 2) if prev_close else None,
            "change_pct": round(change_pct, 2),
            "points_written": len(points)
        }
    except Exception as e:
        print(f"  Price error for {ticker}: {e}")
        return {"ticker": ticker, "price": None, "change_pct": 0}


def collect_all():
    print("=" * 50)
    print("ZelderStock Free Collector")
    print("=" * 50)

    all_results = []

    for ticker in TICKERS:
        print(f"\n{ticker}:")

        print(f"  Fetching Google News RSS...")
        signals = fetch_google_news(ticker)
        print(f"  {len(signals)} articles scored")

        if signals:
            composite = compute_composite(signals)
            avg_score = sum(s["score"] for s in signals) / len(signals)
            influx.write_sentiment(ticker, avg_score, composite, "google_news", len(signals))
            cache_set(f"composite:{ticker}", {"score": composite, "signal_count": len(signals)}, ttl=3600)
            cache_set(f"signals:{ticker}", signals, ttl=3600)
            print(f"  Sentiment: {composite:.1f} ({'bullish' if composite >= 60 else 'bearish' if composite <= 40 else 'neutral'})")
        else:
            composite = 50.0

        print(f"  Fetching Yahoo Finance price...")
        price_data = fetch_price_data(ticker)
        if price_data["price"]:
            print(f"  Price: ${price_data['price']} ({price_data['change_pct']:+.2f}%)")
            cache_set(f"price:{ticker}", price_data, ttl=300)
        
        all_results.append({"ticker": ticker, "composite": composite, **price_data})

    print("\n" + "=" * 50)
    print("Summary:")
    for r in all_results:
        price_str = f"${r['price']}" if r.get("price") else "N/A"
        print(f"  {r['ticker']}: sentiment={r['composite']:.1f} price={price_str} ({r.get('change_pct', 0):+.2f}%)")
    print("=" * 50)


if __name__ == "__main__":
    collect_all()