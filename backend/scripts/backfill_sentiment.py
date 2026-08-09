"""Backfill historical daily news sentiment from GDELT DOC 2.0 into InfluxDB.

The live collector only sees the last ~24h of news, so the LSTM's sentiment input
could never be evaluated over more than ~90 days. GDELT provides free, keyless
news sentiment history back to 2015.

Implementation: one ``mode=timelinetone`` request per ticker returns GDELT's
*Average Tone* series across the entire requested window at daily resolution
(artlist mode proved unusable here — large spans get throttled to 429 and the
Tone field is stripped from responses). Each daily value is the mean compound
tone of that day's articles and is mapped to the same 0-100 composite scale as
the live collector, then written as one ``sentiment`` point per day
(tag source="gdelt") so fetch_live_daily_context() picks it up exactly like
live data.

Rate limits: the free tier throttles artlist/timeline queries (HTTP 429).
The script spaces tickers by ~3 minutes (tune with --sleep) and retries 429s
with an escalating backoff, so an interrupted run can simply be re-run later
(it re-queries and rewrites the same (ticker, day) points — idempotent).

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
SLEEP_SECONDS = 180.0
RETRY_BACKOFFS = [120.0, 240.0, 360.0, 480.0, 600.0]


def tone_to_composite(value: float) -> float:
    """Map a GDELT average tone (-100..100, usually small) to the 0-100 composite scale.

    The live collector uses composite = 50 + avg_score * 50 with avg_score in
    [-1, 1]; GDELT tone is that avg_score scaled by 100.
    """
    score = max(-1.0, min(1.0, value / 100.0))
    return round(50.0 + score * 50.0, 1)


def fetch_tone_timeline(query: str, start: datetime, end: datetime) -> list[dict]:
    """Return [{day: 'YYYYMMDD', value: float}] for the Average Tone series."""
    params = {
        "query": query,
        "mode": "timelinetone",
        "format": "json",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    for attempt in range(len(RETRY_BACKOFFS) + 1):
        try:
            resp = httpx.get(GDELT_URL, params=params, timeout=120)
            if resp.status_code == 429:
                wait = RETRY_BACKOFFS[attempt] if attempt < len(RETRY_BACKOFFS) else RETRY_BACKOFFS[-1]
                print(f"    429, backing off {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            for series in data.get("timeline", []):
                if series.get("series") != "Average Tone":
                    continue
                out = []
                for point in series.get("data", []):
                    date = point.get("date", "")
                    value = point.get("value")
                    if len(date) >= 8 and value is not None:
                        out.append({"day": date[:8], "value": float(value)})
                return out
            return []
        except (httpx.HTTPError, ValueError) as e:
            print(f"    request error: {e}; retrying", flush=True)
            wait = RETRY_BACKOFFS[attempt] if attempt < len(RETRY_BACKOFFS) else RETRY_BACKOFFS[-1]
            time.sleep(wait)
    return []


def write_sentiment_day(client, ticker: str, day: str, composite: float):
    ts = datetime.strptime(day, "%Y%m%d").replace(tzinfo=UTC).replace(hour=12)
    score = (composite - 50.0) / 50.0
    point = (
        Point("sentiment")
        .tag("ticker", ticker)
        .tag("source", "gdelt")
        .field("score", round(score, 4))
        .field("composite", composite)
        .field("signal_count", 1.0)
        .time(ts)
    )
    client.write_api(write_options=SYNCHRONOUS).write(bucket=settings.influx_bucket, record=point)


def backfill(tickers: list[str], years: int, dry_run: bool = False, sleep: float = SLEEP_SECONDS) -> int:
    client = get_influx_client()
    end = datetime.now(UTC)
    start = end - timedelta(days=365 * years)

    total_points = 0
    for ticker in tickers:
        print(f"{ticker}:", flush=True)
        points = fetch_tone_timeline(ticker, start, end)
        if not points:
            print(f"  no timeline data returned", flush=True)
            continue
        if dry_run:
            print(f"  {len(points)} daily tone values "
                  f"({points[0]['day']}..{points[-1]['day']}), nothing written", flush=True)
        else:
            for p in points:
                write_sentiment_day(client, ticker, p["day"], tone_to_composite(p["value"]))
            print(f"  wrote {len(points)} daily sentiment points "
                  f"({points[0]['day']}..{points[-1]['day']})", flush=True)
        total_points += len(points)
        time.sleep(sleep)
    client.close()
    print(f"\nTotal: {total_points} daily sentiment points"
          f" {'(dry run, nothing written)' if dry_run else 'written to InfluxDB'}")
    return total_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical daily news sentiment from GDELT")
    parser.add_argument("--tickers", default=",".join(TRACKED_TICKERS),
                        help="comma-separated tickers (default: all tracked)")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=SLEEP_SECONDS,
                        help="seconds between tickers (default %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="count tone values without writing")
    args = parser.parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    backfill(tickers, args.years, args.dry_run, args.sleep)
