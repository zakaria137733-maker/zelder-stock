import logging
from datetime import UTC, datetime

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.client.write_api import SYNCHRONOUS

from config import settings

logger = logging.getLogger(__name__)

_client: InfluxDBClient | None = None


def get_influx_client() -> InfluxDBClient:
    """Factory for a fresh Influx client, refusing to connect without a token.

    For one-shot batch paths (seed / backfill / training scripts) that close
    their own client. Long-lived serving paths must use get_client() instead so
    the connection is reused across requests instead of opened per request.
    """
    if not settings.influx_token:
        raise RuntimeError(
            "INFLUX_TOKEN is not set in the environment or .env — refusing to connect without a token"
        )
    return InfluxDBClient(
        url=settings.influx_url,
        token=settings.influx_token,
        org=settings.influx_org,
    )


def get_client() -> InfluxDBClient:
    """The module-level singleton for the API/serving path.

    Created once and reused for the lifetime of the process. Do NOT close it in
    per-request finally blocks — that defeats the caching. One-shot scripts
    should use get_influx_client() and close their own instance.
    """
    global _client
    if _client is None:
        _client = get_influx_client()
    return _client


def ensure_buckets() -> list[str]:
    """Create the app's InfluxDB buckets if missing (best-effort, non-fatal).

    A fresh docker-compose instance only initializes the `stock_trades` bucket
    (DOCKER_INFLUXDB_INIT_BUCKET); the sentiment bucket is created here at
    startup so writes never silently fail against a first-boot database.
    Returns the names of buckets that were created.
    """
    wanted = {settings.influx_bucket, settings.influx_trades_bucket}
    created = []
    try:
        client = get_client()
        existing = {b.name for b in client.buckets_api().find_buckets().buckets}
        for name in sorted(wanted - existing):
            client.buckets_api().create_bucket(bucket_name=name, org=settings.influx_org)
            created.append(name)
            logger.info("Created InfluxDB bucket: %s", name)
    except Exception as e:
        logger.warning("Could not ensure InfluxDB buckets (%s): %s", settings.influx_org, e)
    return created


def _clamp_hours(hours: int) -> int:
    return max(1, min(int(hours), 24 * 365))


def _clamp_limit(limit: int) -> int:
    return max(1, min(int(limit), 1000))


def write_sentiment(ticker: str, score: float, composite: float, source: str, signal_count: int = 1):
    client = get_client()
    write_api = client.write_api(write_options=SYNCHRONOUS)
    point = (
        Point("sentiment")
        .tag("ticker", ticker)
        .tag("source", source)
        .field("score", float(score))
        .field("composite", float(composite))
        .field("signal_count", float(signal_count))
        .time(datetime.now(UTC))
    )
    try:
        write_api.write(bucket=settings.influx_bucket, record=point)
    except InfluxDBError as e:
        logger.warning("InfluxDB write error (%s/%s): %s", ticker, source, e)


def write_trade(ticker: str, side: str, price: float, quantity: int, customer_id: str = "", is_demo: bool = False):
    client = get_client()
    write_api = client.write_api(write_options=SYNCHRONOUS)
    point = (
        Point("trades")
        .tag("ticker", ticker)
        .tag("side", side)
        .tag("customer_id", customer_id)
        .field("price", float(price))
        .field("quantity", int(quantity))
        .field("total_usd", float(price * quantity))
        .time(datetime.now(UTC))
    )
    if is_demo:
        point = point.tag("is_demo", "true")
    try:
        write_api.write(bucket=settings.influx_trades_bucket, record=point)
    except InfluxDBError as e:
        logger.warning("InfluxDB trade write error (%s/%s): %s", ticker, side, e)


def query_sentiment_history(ticker: str, hours: int = 24) -> list[dict]:
    client = get_client()
    query_api = client.query_api()
    hours = _clamp_hours(hours)
    flux = (
        "from(bucket: bucket)"
        " |> range(start: duration(v: hours))"
        ' |> filter(fn: (r) => r._measurement == "sentiment" and r.ticker == ticker)'
        ' |> filter(fn: (r) => r._field == "composite")'
        ' |> filter(fn: (r) => r.source != "demo")'
        " |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)"
        ' |> sort(columns: ["_time"])'
    )
    params = {"bucket": settings.influx_bucket, "hours": f"-{hours}h", "ticker": ticker}
    try:
        tables = query_api.query(flux, params=params)
        return [
            {"time": record.get_time().isoformat(), "value": round(record.get_value(), 2)}
            for table in tables
            for record in table.records
        ]
    except Exception as e:
        logger.warning("InfluxDB sentiment history query error (%s): %s", ticker, e)
        return []


def query_recent_trades(ticker: str, limit: int = 20, customer_id: str | None = None) -> list[dict]:
    client = get_client()
    query_api = client.query_api()
    limit = _clamp_limit(limit)
    customer_clause = ' and r.customer_id == customer_id' if customer_id else ""
    params = {"bucket": settings.influx_trades_bucket, "ticker": ticker, "max_n": limit}
    if customer_id:
        params["customer_id"] = customer_id
    flux = (
        "from(bucket: bucket)"
        " |> range(start: -24h)"
        ' |> filter(fn: (r) => r._measurement == "trades" and r.ticker == ticker)'
        ' |> filter(fn: (r) => r._field == "price")'
        f' |> filter(fn: (r) => r.is_demo != "true"{customer_clause})'
        ' |> sort(columns: ["_time"], desc: true)'
        " |> limit(n: max_n)"
    )
    try:
        tables = query_api.query(flux, params=params)
        results = []
        for table in tables:
            for record in table.records:
                results.append({
                    "time": record.get_time().isoformat(),
                    "ticker": record.values.get("ticker"),
                    "side": record.values.get("side"),
                    "price": record.get_value(),
                    "customer_id": record.values.get("customer_id", ""),
                })
        return results
    except Exception as e:
        logger.warning("InfluxDB recent trades query error (%s): %s", ticker, e)
        return []


def query_price_history(ticker: str, hours: int = 24) -> list[dict]:
    client = get_client()
    query_api = client.query_api()
    hours = _clamp_hours(hours)
    flux = (
        "from(bucket: bucket)"
        " |> range(start: duration(v: hours))"
        ' |> filter(fn: (r) => r._measurement == "prices" and r.ticker == ticker and r._field == "close")'
        " |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)"
        ' |> sort(columns: ["_time"])'
    )
    params = {"bucket": settings.influx_bucket, "hours": f"-{hours}h", "ticker": ticker}
    try:
        tables = query_api.query(flux, params=params)
        return [
            {"time": record.get_time().isoformat(), "price": round(record.get_value(), 2)}
            for table in tables for record in table.records
        ]
    except Exception as e:
        logger.warning("InfluxDB price history query error (%s): %s", ticker, e)
        return []
