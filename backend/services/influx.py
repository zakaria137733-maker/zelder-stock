from datetime import UTC, datetime

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.client.write_api import SYNCHRONOUS

from config import settings

_client: InfluxDBClient | None = None


def get_client() -> InfluxDBClient:
    global _client
    if _client is None:
        _client = InfluxDBClient(
            url=settings.influx_url,
            token=settings.influx_token,
            org=settings.influx_org,
        )
    return _client


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


def write_trade(ticker: str, side: str, price: float, quantity: int, customer_id: str = ""):
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
    try:
        write_api.write(bucket="stock_trades", record=point)
    except InfluxDBError as e:
        print(f"InfluxDB write error: {e}")


def query_sentiment_history(ticker: str, hours: int = 24) -> list[dict]:
    client = get_client()
    query_api = client.query_api()
    flux = f"""
        from(bucket: "{settings.influx_bucket}")
          |> range(start: -{hours}h)
          |> filter(fn: (r) => r._measurement == "sentiment" and r.ticker == "{ticker}")
          |> filter(fn: (r) => r._field == "composite")
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
          |> sort(columns: ["_time"])
    """
    try:
        tables = query_api.query(flux)
        return [
            {"time": record.get_time().isoformat(), "value": round(record.get_value(), 2)}
            for table in tables
            for record in table.records
        ]
    except Exception as e:
        print(f"InfluxDB query error: {e}")
        return []


def query_recent_trades(ticker: str, limit: int = 20) -> list[dict]:
    client = get_client()
    query_api = client.query_api()
    flux = f"""
        from(bucket: "stock_trades")
          |> range(start: -24h)
          |> filter(fn: (r) => r._measurement == "trades" and r.ticker == "{ticker}")
          |> filter(fn: (r) => r._field == "price")
          |> sort(columns: ["_time"], desc: true)
          |> limit(n: {limit})
    """
    try:
        tables = query_api.query(flux)
        results = []
        for table in tables:
            for record in table.records:
                results.append({
                    "time": record.get_time().isoformat(),
                    "ticker": record.values.get("ticker"),
                    "side": record.values.get("side"),
                    "price": record.get_value(),
                })
        return results
    except Exception as e:
        print(f"InfluxDB query error: {e}")
        return []


def query_price_history(ticker: str, hours: int = 24) -> list[dict]:
    client = get_client()
    query_api = client.query_api()
    flux = (
        f'from(bucket: "sentiment_scores")'
        f' |> range(start: -{hours}h)'
        f' |> filter(fn: (r) => r._measurement == "prices" and r.ticker == "{ticker}" and r._field == "close")'
        f' |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)'
        f' |> sort(columns: ["_time"])'
    )
    try:
        tables = query_api.query(flux)
        return [
            {"time": record.get_time().isoformat(), "price": round(record.get_value(), 2)}
            for table in tables for record in table.records
        ]
    except Exception as e:
        print(f"Price history error: {e}")
        return []

