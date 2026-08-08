import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yfinance as yf

from services.news_collector import collect_google_sentiment, fetch_google_news
from services.redis_client import cache_set
from tickers import TICKERS


def fetch_price_data(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d", interval="1h")
        info = t.fast_info
        current_price = float(info.last_price) if hasattr(info, "last_price") else None
        prev_close = float(info.previous_close) if hasattr(info, "previous_close") else None
        change_pct = ((current_price - prev_close) / prev_close * 100) if current_price and prev_close else 0

        from influxdb_client import Point
        from influxdb_client.client.write_api import SYNCHRONOUS
        from services.influx import get_influx_client

        client = get_influx_client()
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
    print("SentimentIQ Free Collector")
    print("=" * 50)

    all_results = []

    for ticker in TICKERS:
        print(f"\n{ticker}:")

        print(f"  Fetching Google News RSS...")
        result = collect_google_sentiment(ticker)
        composite = result["composite"]
        print(f"  {result['signal_count']} articles scored")
        print(f"  Sentiment: {composite:.1f} ({'bullish' if composite >= 60 else 'bearish' if composite <= 40 else 'neutral'})")

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