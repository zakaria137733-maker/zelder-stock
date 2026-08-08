"""Backfill historical daily news sentiment from GDELT DOC 2.0 into InfluxDB.

The live collector only sees the last ~24h of news, so the LSTM's sentiment input
could never be evaluated over more than ~90 days. GDELT provides free, keyless
news sentiment history back to 2015. This script queries it per ticker in
date chunks, computes a daily composite on the same 0-100 scale as the live
collector, and writes one `sentiment` point per day (tag source="gdelt") so
fetch_live_daily_context() picks it up exactly like live data.

GDELT asks for >= 1 request per 5 seconds; the script sleeps 5.2s per chunk and
retries 429s with backoff. It is idempotent: re-running overwrites the same
(ticker, day) points, so an interrupted run can simply be re-run.

    docker-compose exec api python scripts/backfill_sentiment.py --tickers AAPL,NVDA
    docker-compose exec api python scripts/backfill_sentiment.py --dry-run
"""

import argparse
import os
import sys
import time
from datetime import UTC, datetime, timedelta

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS

from config import settings
from services.influx import get_influx_client
from tickers import TICKERS as TRACKED_TICKERS

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
SLEEP_SECONDS = 7.0
RETRY_BACKOFFS = [15.0, 30.0, 60.0, 90.0]


def parse_compound(tone: str | None) -> float | None:
    """GDELT Tone field: comma list whose first value is compound sentiment (-100..100)."""
    if not tone:
        return None
    try:
        return float(tone.split(",")[0])
    except (ValueError, IndexError):
        return None


def gdelt_articles(query: str, start: datetime, end: datetime) -> list[dict]:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 250,
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    for attempt in range(len(RETRY_BACKOFFS) + 1):
        try:
            resp = httpx.get(GDELT_URL, params=params, timeout=60)
            if resp.status_code == 429:
                wait = RETRY_BACKOFFS[attempt] if attempt < len(RETRY_BACKOFFS) else RETRY_BACKOFFS[-1]
                print(f"    429, backing off {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            articles = []
            for art in data.get("articles", []):
                day = art.get("seendate", "")[:8]
                compound = parse_compound(art.get("tone"))
                if day and compound is not None:
                    articles.append({"day": day, "compound": compound})
            return articles
        except (httpx.HTTPError, ValueError) as e:
            print(f"    request error: {e}; retrying", flush=True)
            wait = RETRY_BACKOFFS[attempt] if attempt < len(RETRY_BACKOFFS) else RETRY_BACKOFFS[-1]
            time.sleep(wait)
    return []


def daily_composites(articles: list[dict]) -> dict[str, dict]:
    """{day: {composite, signal_count}} on the 0-100 scale used by the live collector."""
    by_day: dict[str, list[float]] = {}
    for a in articles:
        by_day.setdefault(a["day"], []).append(a["compound"])
    out = {}
    for day, compounds in by_day.items():
        score = max(-1.0, min(1.0, sum(compounds) / len(compounds) / 100.0))
        out[day] = {"composite": round(50.0 + score * 50.0, 1), "signal_count": len(compounds)}
    return out


def write_sentiment_day(client, ticker: str, day: str, entry: dict):
    ts = datetime.strptime(day, "%Y%m%d").replace(tzinfo=UTC).replace(hour=12)
    score = (entry["composite"] - 50.0) / 50.0
    point = (
        Point("sentiment")
        .tag("ticker", ticker)
        .tag("source", "gdelt")
        .field("score", round(score, 4))
        .field("composite", entry["composite"])
        .field("signal_count", float(entry["signal_count"]))
        .time(ts)
    )
    client.write_api(write_options=SYNCHRONOUS).write(bucket=settings.influx_bucket, record=point)


def backfill(tickers: list[str], years: int, chunk_days: int, dry_run: bool = False) -> int:
    client = get_influx_client()
    end = datetime.now(UTC)
    start = end - timedelta(days=365 * years)
    chunk = timedelta(days=chunk_days)

    total_points = 0
    for ticker in tickers:
        print(f"{ticker}:")
        cursor = start
        chunk_no = 0
        while cursor < end:
            chunk_end = min(cursor + chunk, end)
            articles = gdelt_articles(ticker, cursor, chunk_end)
            comps = daily_composites(articles)
            if comps:
                if dry_run:
                    print(f"  chunk {chunk_no} ({cursor.date()}..{chunk_end.date()}): "
                          f"{len(articles)} articles, {len(comps)} days", flush=True)
                else:
                    for day, entry in comps.items():
                        write_sentiment_day(client, ticker, day, entry)
                total_points += len(comps)
            else:
                print(f"  chunk {chunk_no} ({cursor.date()}..{chunk_end.date()}): 0 articles", flush=True)
            cursor = chunk_end
            chunk_no += 1
            time.sleep(SLEEP_SECONDS)
        print(f"  done: {total_points} daily sentiment points so far", flush=True)
    client.close()
    print(f"\nTotal: {total_points} daily sentiment points"
          f" {'(dry run, nothing written)' if dry_run else 'written to InfluxDB'}")
    return total_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical daily news sentiment from GDELT")
    parser.add_argument("--tickers", default=",".join(TRACKED_TICKERS),
                        help="comma-separated tickers (default: all tracked)")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--chunk-days", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true", help="count articles/days without writing")
    args = parser.parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    backfill(tickers, args.years, args.chunk_days, args.dry_run)
