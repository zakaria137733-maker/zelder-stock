#!/usr/bin/env python3
"""One-command offline demo seed for ZelderStock.

Wipes (default) and reseeds both Influx buckets plus MongoDB with a complete,
deterministic demo dataset so the whole product renders without any live
collector, Yahoo Finance, NewsAPI, or cloud dependencies:

* daily prices in both ``prices_daily`` and ``prices`` (close/volume),
  anchored on real recent closes and backdated ``--days`` days (default 40);
* daily sentiment (backdated) plus hourly sentiment for the last 48h, tagged
  ``source="backfill"`` so live queries (which filter ``source != "demo"``) show it;
* daily ``market_index`` series for SPY / QQQ / VIX;
* ~200 trades anchored on real closes, unmarked (no ``is_demo`` tag) so live
  trade queries render them, spread across the seeded Mongo customers;
* Mongo customers including ``demo@sentimentiq.io`` / ``DemoPass123!``;
* a pre-warmed Redis cache (``composite:{ticker}``, ``price:{ticker}``,
  ``signals:{ticker}``, ``news:{ticker}``) so ``/api/sentiment/AAPL`` answers
  instantly offline (composite 72 for AAPL);

and finishes with a live self-check summary (Mongo / Influx / Redis counts).

Deterministic: ``random.seed(42)`` at the top means every run produces the same
series (closes, composites, trade prices) for the same ``--days``.

Usage (from the repo root, inside the API container):
    docker compose exec -T api python scripts/seed_demo.py
    docker compose exec -T api python scripts/seed_demo.py --days 90
    docker compose exec -T api python scripts/seed_demo.py --no-wipe
"""

import argparse
import math
import os
import random
import sys
import time
from datetime import UTC, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from config import settings
from services.auth import hash_password
from services.mongo import get_db
from services.redis_client import cache_get, cache_set
from tickers import TICKERS

BUCKETS = [settings.influx_bucket, settings.influx_trades_bucket]
HOURS_BACK = 48
N_TRADES = 200
DEMO_EMAIL = "demo@sentimentiq.io"
DEMO_PASSWORD = "DemoPass123!"

BASE_SCORES = {"AAPL": 72, "TSLA": 41, "NVDA": 88, "MSFT": 68, "GOOGL": 75, "AMZN": 63, "META": 58}
ANCHOR_CLOSES = {"AAPL": 189.42, "TSLA": 248.17, "NVDA": 876.54, "MSFT": 415.22, "GOOGL": 175.83, "AMZN": 185.60, "META": 490.32}
MARKET_ANCHORS = {"SPY": 580.0, "QQQ": 505.0, "VIX": 18.0}
MARKET_VOL = {"SPY": 0.012, "QQQ": 0.013, "VIX": 0.035}

CUSTOMERS = [
    {"name": "Sarah Chen", "email": "sarah.chen@example.com", "portfolio_value": 142800, "watchlist": ["AAPL", "NVDA"], "risk_profile": "conservative", "sentiment_score": 87},
    {"name": "Marcus Webb", "email": "marcus.webb@example.com", "portfolio_value": 58200, "watchlist": ["TSLA", "AAPL"], "risk_profile": "aggressive", "sentiment_score": 72},
    {"name": "Priya Nair", "email": "priya.nair@example.com", "portfolio_value": 214500, "watchlist": ["NVDA", "MSFT", "GOOGL"], "risk_profile": "moderate", "sentiment_score": 91},
    {"name": "James O'Brien", "email": "james.obrien@example.com", "portfolio_value": 31000, "watchlist": ["TSLA"], "risk_profile": "aggressive", "sentiment_score": 38},
    {"name": "Yuki Tanaka", "email": "yuki.tanaka@example.com", "portfolio_value": 98700, "watchlist": ["AAPL", "AMZN", "META"], "risk_profile": "moderate", "sentiment_score": 65},
    {"name": "Alex Rivera", "email": "alex.rivera@example.com", "portfolio_value": 175300, "watchlist": ["NVDA", "MSFT"], "risk_profile": "conservative", "sentiment_score": 79},
    {"name": "Dana Kim", "email": "dana.kim@example.com", "portfolio_value": 22400, "watchlist": ["TSLA", "AMZN"], "risk_profile": "aggressive", "sentiment_score": 44},
    {"name": "Omar Hassan", "email": "omar.hassan@example.com", "portfolio_value": 310000, "watchlist": ["AAPL", "MSFT", "GOOGL", "NVDA"], "risk_profile": "conservative", "sentiment_score": 83},
]

DEMO_USER = {
    "name": "Demo User",
    "email": DEMO_EMAIL,
    "password": DEMO_PASSWORD,
    "portfolio_value": 100000,
    "watchlist": ["AAPL", "NVDA", "MSFT"],
    "risk_profile": "moderate",
    "sentiment_score": 72,
}

SIGNAL_TEMPLATES = [
    (0.42, "{ticker} bullish momentum continues as analysts lift targets"),
    (0.31, "{ticker} sees heavy institutional buying after strong earnings"),
    (-0.18, "Analysts stay cautious on {ticker} valuation despite rally"),
    (0.22, "{ticker} volume surges amid renewed retail interest"),
    (0.05, "{ticker} faces mixed sentiment heading into next week"),
]

NOW = datetime.now(UTC)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _day_ts(days_ago: int) -> datetime:
    return (NOW - timedelta(days=days_ago)).replace(hour=12, minute=0, second=0, microsecond=0)


def _random_walk(days: int, anchor: float, vol: float) -> list[float]:
    """Return `days` daily closes ending exactly at `anchor` (today's real close)."""
    steps = [math.exp(random.gauss(0, vol)) for _ in range(days)]
    closes = [0.0] * days
    cur = anchor
    for i in range(days - 1, -1, -1):
        closes[i] = cur
        cur = cur / steps[i]
    return closes


def _volumes(days: int, lo: float, hi: float) -> list[float]:
    return [round(random.uniform(lo, hi), 0) for _ in range(days)]


def _sentiment_point(ticker: str, composite: float, ts: datetime) -> Point:
    return (
        Point("sentiment")
        .tag("ticker", ticker)
        .tag("source", "backfill")
        .field("composite", round(composite, 2))
        .field("score", round((composite - 50) / 50, 4))
        .field("signal_count", float(random.randint(3, 20)))
        .time(ts)
    )


def ensure_and_wipe(client, wipe: bool) -> None:
    """Ensure both buckets exist; wipe them unless `wipe` is False (--no-wipe)."""
    buckets_api = client.buckets_api()
    existing = {b.name for b in buckets_api.find_buckets().buckets}
    for name in BUCKETS:
        if name not in existing:
            buckets_api.create_bucket(bucket_name=name, org=settings.influx_org)
            print(f"[influx] created bucket: {name}")

    if not wipe:
        print("[influx] --no-wipe: leaving existing data in place")
        return

    wipe_plan = {
        settings.influx_bucket: ["sentiment", "prices", "prices_daily", "market_index"],
        settings.influx_trades_bucket: ["trades"],
    }
    start = datetime(1970, 1, 1, tzinfo=UTC)
    stop = datetime.now(UTC)
    delete_api = client.delete_api()
    for bucket, measurements in wipe_plan.items():
        for measurement in measurements:
            for attempt in range(3):
                try:
                    delete_api.delete(start, stop, f'_measurement="{measurement}"', bucket=bucket, org=settings.influx_org)
                    break
                except Exception as e:
                    print(f"[influx] wipe error ({bucket}/{measurement}, attempt {attempt + 1}): {e}")
                    if attempt == 2:
                        print(f"[influx] giving up on {bucket}/{measurement}; continuing")
                    else:
                        time.sleep(5)
        print(f"[influx] wiped bucket: {bucket}")


def count_points(client, bucket: str, measurement: str, field: str) -> int:
    flux = (
        "from(bucket: bucket)"
        " |> range(start: 0)"
        ' |> filter(fn: (r) => r._measurement == measurement)'
        ' |> filter(fn: (r) => r._field == field)'
        " |> count()"
    )
    params = {"bucket": bucket, "measurement": measurement, "field": field}
    try:
        tables = client.query_api().query(flux, params=params)
        return sum(int(record.get_value()) for table in tables for record in table.records)
    except Exception as e:
        print(f"[self-check] count error ({bucket}/{measurement}): {e}")
        return -1


async def seed_customers(wipe: bool) -> tuple[list[dict], str]:
    """Seed the 8 demo customers + demo@sentimentiq.io.

    Returns (credentials, demo_customer_id). With wipe, the collection is
    dropped and reseeded; otherwise only missing customers are inserted and the
    demo user's password/profile are refreshed (upsert semantics).
    """
    db = get_db()
    if wipe:
        await db.customers.drop()
        print("[mongo] dropped customers collection")
    await db.customers.create_index("email", unique=True)

    credentials = []
    demo_id = None

    async def _upsert(doc: dict, password: str) -> str:
        existing = await db.customers.find_one({"email": doc["email"]})
        if existing:
            await db.customers.update_one(
                {"_id": existing["_id"]},
                {"$set": {"password_hash": hash_password(password)}},
            )
            return str(existing["_id"])
        result = await db.customers.insert_one(doc)
        return str(result.inserted_id)

    for c in CUSTOMERS:
        password = f"Demo{random.randint(100000, 999999)}!"
        doc = dict(c)
        doc["password_hash"] = hash_password(password)
        doc["created_at"] = NOW - timedelta(days=random.randint(30, 365))
        cid = await _upsert(doc, password)
        credentials.append({"email": c["email"], "password": password, "id": cid})

    demo_doc = dict(DEMO_USER)
    demo_doc["password_hash"] = hash_password(DEMO_PASSWORD)
    demo_doc["created_at"] = NOW - timedelta(days=90)
    demo_id = await _upsert(demo_doc, DEMO_PASSWORD)
    credentials.append({"email": DEMO_EMAIL, "password": DEMO_PASSWORD, "id": demo_id})

    print(f"[mongo] seeded {len(credentials)} customers (demo: {DEMO_EMAIL} / {DEMO_PASSWORD})")
    return credentials, demo_id


def seed_prices(client, days: int) -> dict[str, list[float]]:
    """Seed daily closes/volumes into prices_daily + prices for every ticker."""
    daily_points, live_points = [], []
    closes_map: dict[str, list[float]] = {}
    for ticker in TICKERS:
        closes = _random_walk(days, ANCHOR_CLOSES[ticker], 0.016)
        vols = _volumes(days, 1.5e6, 3.0e7)
        closes_map[ticker] = closes
        for i in range(days):
            ts = _day_ts(days - 1 - i)
            daily_points.append(
                Point("prices_daily")
                .tag("ticker", ticker)
                .field("close", round(closes[i], 2))
                .field("volume", vols[i])
                .time(ts)
            )
            live_ts = (NOW.replace(minute=0, second=0, microsecond=0)) if i == days - 1 else ts
            live_points.append(
                Point("prices")
                .tag("ticker", ticker)
                .field("close", round(closes[i], 2))
                .field("volume", vols[i])
                .time(live_ts)
            )
    write_api = client.write_api(write_options=SYNCHRONOUS)
    write_api.write(bucket=settings.influx_bucket, record=daily_points)
    write_api.write(bucket=settings.influx_bucket, record=live_points)
    print(f"[influx] seeded {len(TICKERS) * days} prices_daily + prices points")
    return closes_map


def seed_market_index(client, days: int) -> None:
    points = []
    for name, anchor in MARKET_ANCHORS.items():
        closes = _random_walk(days, anchor, MARKET_VOL[name])
        vols = _volumes(days, 4.0e7, 9.0e7)
        for i in range(days):
            close = _clamp(closes[i], 10.0, 40.0) if name == "VIX" else round(closes[i], 2)
            points.append(
                Point("market_index")
                .tag("ticker", name)
                .field("close", close)
                .field("volume", vols[i])
                .time(_day_ts(days - 1 - i))
            )
    write_api = client.write_api(write_options=SYNCHRONOUS)
    write_api.write(bucket=settings.influx_bucket, record=points)
    print(f"[influx] seeded {len(MARKET_ANCHORS) * days} market_index points")


def seed_sentiment(client, days: int) -> None:
    """Daily sentiment (backdated) + hourly for the last 48h, source='backfill'."""
    points = []
    daily_days = max(0, days - 2)
    for ticker in TICKERS:
        base = BASE_SCORES[ticker]
        for i in range(daily_days):
            comp = _clamp(base + random.gauss(0, 4), 10, 95)
            points.append(_sentiment_point(ticker, comp, _day_ts(days - 1 - i)))
        for h in range(HOURS_BACK, 0, -1):
            comp = _clamp(base + random.gauss(0, 4), 10, 95)
            points.append(_sentiment_point(ticker, comp, NOW - timedelta(hours=h)))
    write_api = client.write_api(write_options=SYNCHRONOUS)
    write_api.write(bucket=settings.influx_bucket, record=points)
    print(f"[influx] seeded {len(points)} sentiment points ({daily_days} daily + {HOURS_BACK} hourly per ticker)")


def seed_trades(client, customer_ids: list[str], demo_id: str) -> None:
    """~200 trades anchored on real closes, unmarked so live queries render them."""
    points = []
    for _ in range(N_TRADES):
        ticker = random.choice(TICKERS)
        side = random.choice(["buy", "sell"])
        price = round(ANCHOR_CLOSES[ticker] * (1 + random.uniform(-0.01, 0.01)), 2)
        qty = random.randint(5, 150)
        cust = demo_id if random.random() < 0.4 else random.choice(customer_ids)
        points.append(
            Point("trades")
            .tag("ticker", ticker)
            .tag("side", side)
            .tag("customer_id", cust)
            .field("price", price)
            .field("quantity", qty)
            .field("total_usd", round(price * qty, 2))
            .time(NOW - timedelta(hours=random.uniform(0, 24)))
        )
    write_api = client.write_api(write_options=SYNCHRONOUS)
    write_api.write(bucket=settings.influx_trades_bucket, record=points)
    print(f"[influx] seeded {N_TRADES} trades (unmarked, no is_demo tag)")


def _demo_signals(ticker: str) -> list[dict]:
    signals = []
    for i, (score, template) in enumerate(SIGNAL_TEMPLATES):
        age = i * 4 + 1.0
        label = "positive" if score >= 0.1 else "negative" if score <= -0.1 else "neutral"
        signals.append({
            "ticker": ticker,
            "source": "google_news",
            "source_name": "ZelderStock Demo Feed",
            "title": template.format(ticker=ticker),
            "url": f"https://example.invalid/siq-demo/{ticker.lower()}-{i}",
            "published_at": (NOW - timedelta(hours=age)).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "age_hours": round(age, 2),
            "score": score,
            "label": label,
            "confidence": 0.7,
        })
    return signals


def prewarm_redis(closes_map: dict[str, list[float]]) -> None:
    failures = []
    for ticker in TICKERS:
        base = BASE_SCORES[ticker]
        closes = closes_map[ticker]
        change_pct = round((closes[-1] - closes[0]) / closes[0] * 100, 2)
        signals = _demo_signals(ticker)
        for key, value, ttl in (
            (f"composite:{ticker}", {"score": base, "signal_count": 12}, 86400),
            (f"price:{ticker}", {"ticker": ticker, "price": round(closes[-1], 2), "change_pct": change_pct}, 86400),
            (f"signals:{ticker}", signals, 3600),
            (f"news:{ticker}", signals, 3600),
        ):
            try:
                cache_set(key, value, ttl)
            except Exception as e:
                failures.append(f"{key}: {e}")
    if failures:
        print(f"[redis] {len(failures)} prewarm failures: {failures[:3]}...")
    else:
        print(f"[redis] pre-warmed composite/price/signals/news for {len(TICKERS)} tickers")


async def self_check(client, days: int) -> dict:
    print("\n=== Self-check (live) ===")
    db = get_db()
    customer_count = await db.customers.count_documents({})
    demo = await db.customers.find_one({"email": DEMO_EMAIL})

    checks = {}
    checks["mongo.customers"] = customer_count == 9
    checks["mongo.demo_user"] = demo is not None and bool(demo.get("password_hash"))

    senti = count_points(client, settings.influx_bucket, "sentiment", "composite")
    prices_daily = count_points(client, settings.influx_bucket, "prices_daily", "close")
    prices = count_points(client, settings.influx_bucket, "prices", "close")
    market = count_points(client, settings.influx_bucket, "market_index", "close")
    trades = count_points(client, settings.influx_trades_bucket, "trades", "price")

    checks["influx.sentiment"] = senti == 7 * max(0, days - 2) + 7 * HOURS_BACK
    checks["influx.prices_daily"] = prices_daily == 7 * days
    checks["influx.prices"] = prices == 7 * days
    checks["influx.market_index"] = market == 3 * days
    checks["influx.trades"] = trades == N_TRADES

    composite_aapl = cache_get("composite:AAPL")
    checks["redis.composite:AAPL"] = bool(composite_aapl) and float(composite_aapl.get("score", 0)) == BASE_SCORES["AAPL"]
    price_aapl = cache_get("price:AAPL")
    checks["redis.price:AAPL"] = bool(price_aapl) and float(price_aapl.get("price", 0)) == ANCHOR_CLOSES["AAPL"]

    print(f"  mongo.customers      : {customer_count} -> {'OK' if checks['mongo.customers'] else 'FAIL'}")
    print(f"  mongo.demo_user      : {DEMO_EMAIL} hash present -> {'OK' if checks['mongo.demo_user'] else 'FAIL'}")
    print(f"  influx.sentiment     : {senti} -> {'OK' if checks['influx.sentiment'] else 'FAIL'}")
    print(f"  influx.prices_daily  : {prices_daily} -> {'OK' if checks['influx.prices_daily'] else 'FAIL'}")
    print(f"  influx.prices        : {prices} -> {'OK' if checks['influx.prices'] else 'FAIL'}")
    print(f"  influx.market_index  : {market} -> {'OK' if checks['influx.market_index'] else 'FAIL'}")
    print(f"  influx.trades        : {trades} -> {'OK' if checks['influx.trades'] else 'FAIL'}")
    print(f"  redis.composite:AAPL : {composite_aapl} -> {'OK' if checks['redis.composite:AAPL'] else 'FAIL'}")
    print(f"  redis.price:AAPL     : {price_aapl} -> {'OK' if checks['redis.price:AAPL'] else 'FAIL'}")

    return checks


async def main() -> None:
    parser = argparse.ArgumentParser(description="One-command offline demo seed for ZelderStock")
    parser.add_argument("--no-wipe", action="store_true", help="Do not wipe Influx buckets or drop customers (upsert-only)")
    parser.add_argument("--days", type=int, default=40, help="Days of backdated daily data (default 40)")
    args = parser.parse_args()

    days = max(14, args.days)
    wipe = not args.no_wipe
    random.seed(42)

    print("=== ZelderStock demo seed ===")
    print(f"wipe={wipe} days={days}")

    client = InfluxDBClient(
        url=settings.influx_url,
        token=settings.influx_token,
        org=settings.influx_org,
        timeout=600_000,
    )
    try:
        ensure_and_wipe(client, wipe)
        credentials, demo_id = await seed_customers(wipe)
        closes_map = seed_prices(client, days)
        seed_market_index(client, days)
        seed_sentiment(client, days)
        seed_trades(client, [c["id"] for c in credentials], demo_id)
        prewarm_redis(closes_map)

        checks = await self_check(client, days)
        passed = sum(checks.values())
        total = len(checks)
        print(f"\nSelf-check: {passed}/{total} passed")

        print("\nDemo credentials")
        print("  admin   : admin / admin123            -> POST /api/admin/login")
        print(f"  customer: {DEMO_EMAIL} / {DEMO_PASSWORD} -> POST /api/auth/login")
        print("\nURLs")
        print("  API        : http://localhost:8000  (Swagger: http://localhost:8000/docs)")
        print("  InfluxDB   : http://localhost:8086")
        print("  Frontend   : http://localhost:3000")
        print("\nVerification")
        print("  curl -X POST http://localhost:8000/api/admin/login -H 'Content-Type: application/json' -d '{\"username\":\"admin\",\"password\":\"admin123\"}'")
        print("  curl http://localhost:8000/api/sentiment/AAPL")
        print("  curl http://localhost:8000/api/predictions/AAPL -H 'Authorization: Bearer <customer-token>'")
    finally:
        client.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
