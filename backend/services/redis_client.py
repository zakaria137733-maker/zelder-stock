import hashlib
import json

import redis as redis_lib

from config import settings

_client: redis_lib.Redis | None = None


def get_client() -> redis_lib.Redis:
    global _client
    if _client is None:
        _client = redis_lib.from_url(settings.redis_url, decode_responses=True)
    return _client


def cache_set(key: str, value: dict | list, ttl: int = 1800):
    get_client().setex(key, ttl, json.dumps(value))


def cache_get(key: str) -> dict | list | None:
    raw = get_client().get(key)
    return json.loads(raw) if raw else None


def publish_signal(signal: dict):
    get_client().publish("signals", json.dumps(signal))


def add_to_dedup_set(url: str, ttl: int = 86400) -> bool:
    """Returns True if the URL is new (not seen before)."""
    key = "seen:" + hashlib.sha256(url.encode()).hexdigest()
    result = get_client().set(key, 1, ex=ttl, nx=True)
    return result is True
