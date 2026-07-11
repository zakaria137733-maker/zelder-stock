from fastapi import APIRouter
from services.redis_client import cache_get
from influxdb_client import InfluxDBClient
from config import settings

router = APIRouter(prefix="/api/prices", tags=["prices"])
TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]


@router.get("/")
async def get_all_prices():
    results = []
    for ticker in TICKERS:
        cached = cache_get(f"price:{ticker}") or {"ticker": ticker, "price": None, "change_pct": 0}
        results.append(cached)
    return results


@router.get("/{ticker}/history")
async def get_price_history(ticker: str, hours: int = 24):
    ticker = ticker.upper()
    client = InfluxDBClient(url=settings.influx_url, token=settings.influx_token, org=settings.influx_org)
    query_api = client.query_api()
    flux = f"""
        from(bucket: "sentiment_scores")
          |> range(start: -7d)
          |> filter(fn: (r) => r._measurement == "prices" and r.ticker == "{ticker}" and r._field == "close")
          |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
          |> sort(columns: ["_time"])
    """
    try:
        tables = query_api.query(flux)
        return [
            {"time": record.get_time().isoformat(), "price": round(record.get_value(), 2)}
            for table in tables for record in table.records
        ]
    except Exception as e:
        print(f"Price history error: {e}")
        return []
    finally:
        client.close()


@router.get("/{ticker}")
async def get_price(ticker: str):
    ticker = ticker.upper()
    cached = cache_get(f"price:{ticker}")
    if cached:
        return cached
    return {"ticker": ticker, "price": None, "change_pct": 0}