from datetime import UTC, datetime

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.client.write_api import SYNCHRONOUS

from config import settings

_client: InfluxDBClient | None = None


def get_influx_client() -> InfluxDBClient:
    """Single factory for Influx clients, refusing to connect without a token."""
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
    global _client
    if _client is None:
        _client = get_influx_client()
    return _client


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
        print(f"InfluxDB write error: {e}")


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
        write_api.write(bucket="stock_trades", record=point)
    except InfluxDBError as e:
        print(f"InfluxDB write error: {e}")


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
        print(f"InfluxDB query error: {e}")
        return []


def query_recent_trades(ticker: str, limit: int = 20, customer_id: str | None = None) -> list[dict]:
    client = get_client()
    query_api = client.query_api()
    limit = _clamp_limit(limit)
    customer_clause = ' and r.customer_id == customer_id' if customer_id else ""
    params = {"bucket": "stock_trades", "ticker": ticker, "max_n": limit}
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
        print(f"InfluxDB query error: {e}")
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
        print(f"Price history error: {e}")
        return []

