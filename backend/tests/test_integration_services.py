"""Integration tests against the real backing services.

These exercise the production service modules (services/mongo.py, redis_client.py,
influx.py) end-to-end against live MongoDB / Redis / InfluxDB. They auto-skip when
the services are not reachable, so a plain `pytest` run stays offline-friendly, and
they run for real in CI (which starts the services) and locally with `docker-compose up`.
"""

import asyncio
import time

import pytest

from config import settings

pytestmark = pytest.mark.integration


def _services_up() -> bool:
    try:
        from services import influx, redis_client

        async def mongo_up() -> bool:
            from motor.motor_asyncio import AsyncIOMotorClient
            client = AsyncIOMotorClient(settings.mongo_uri, serverSelectionTimeoutMS=2000)
            try:
                await client.admin.command("ping")
                return True
            finally:
                client.close()

        if not influx.get_client().ping():
            return False
        redis_client.get_client().ping()
        return asyncio.run(mongo_up())
    except Exception:
        return False


@pytest.fixture(scope="module")
def services_available():
    for _ in range(40):
        if _services_up():
            break
        time.sleep(1)
    else:
        pytest.skip("Mongo/Influx/Redis not reachable — start them with `docker-compose up -d`")
    from services import influx
    influx.ensure_buckets()


def _poll(fn, attempts=20, delay=0.5):
    result = fn()
    for _ in range(attempts):
        if result:
            return result
        time.sleep(delay)
        result = fn()
    return result


def test_redis_cache_roundtrip(services_available):  # noqa: ARG001 - fixture gates on service availability
    from services import redis_client
    redis_client.get_client().flushdb()
    redis_client.cache_set("it:cache", {"ticker": "AAPL", "score": 71.5}, ttl=120)
    assert redis_client.cache_get("it:cache") == {"ticker": "AAPL", "score": 71.5}
    assert redis_client.cache_get("it:missing") is None


def test_redis_dedup_set(services_available):  # noqa: ARG001 - fixture gates on service availability
    from services import redis_client
    url = "https://example.com/signal/it-dedup"
    assert redis_client.add_to_dedup_set(url, ttl=120) is True
    assert redis_client.add_to_dedup_set(url, ttl=120) is False


def test_mongo_crud_roundtrip(services_available):  # noqa: ARG001 - fixture gates on service availability
    from motor.motor_asyncio import AsyncIOMotorClient
    client = AsyncIOMotorClient(settings.mongo_uri, serverSelectionTimeoutMS=3000)
    col = client.sentimentiq["it_customers"]

    async def run():
        await col.delete_many({})
        await col.insert_one({"_id": "it-1", "name": "Integration User", "email": "it@example.com", "role": "analyst"})
        doc = await col.find_one({"_id": "it-1"})
        assert doc and doc["role"] == "analyst"
        await col.update_one({"_id": "it-1"}, {"$set": {"role": "admin"}})
        doc = await col.find_one({"_id": "it-1"})
        assert doc and doc["role"] == "admin"
        await col.delete_one({"_id": "it-1"})
        assert await col.find_one({"_id": "it-1"}) is None

    asyncio.run(run())
    client.close()


def test_influx_sentiment_roundtrip(services_available):  # noqa: ARG001 - fixture gates on service availability
    from services import influx
    influx.write_sentiment("ITEST", 70.0, 66.0, "integration")
    rows = _poll(lambda: influx.query_sentiment_history("ITEST", hours=48))
    assert rows, "written sentiment row never became queryable"
    assert rows[-1]["value"] == 66.0


def test_influx_trades_roundtrip(services_available):  # noqa: ARG001 - fixture gates on service availability
    from services import influx
    influx.write_trade("ITEST", "BUY", 150.0, 4)
    rows = _poll(lambda: influx.query_recent_trades("ITEST", limit=50))
    assert rows, "written trade never became queryable"
    assert any(r["ticker"] == "ITEST" and r["price"] == 150.0 for r in rows)
