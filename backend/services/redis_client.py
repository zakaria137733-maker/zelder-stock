import hashlib
import json
import logging
import time
from collections import defaultdict, deque

import redis as redis_lib

from config import settings

logger = logging.getLogger(__name__)

_client: redis_lib.Redis | None = None

# In-process sliding-window buckets used only when Redis is unreachable.
_fallback_buckets: dict[str, deque] = defaultdict(deque)

# Throttle warnings so a Redis outage logs once per minute per operation
# instead of spamming one line per request.
_last_logged: dict[str, float] = {}


def _log_down(op: str, exc: Exception):
    now = time.monotonic()
    if now - _last_logged.get(op, 0.0) > 60:
        _last_logged[op] = now
        logger.warning("Redis unavailable (%s), degrading gracefully: %s", op, exc)


def get_client() -> redis_lib.Redis:
    global _client
    if _client is None:
        _client = redis_lib.from_url(settings.redis_url, decode_responses=True)
    return _client


def rate_limit_exceeded(scope: str, identity: str, limit: int, window: int) -> bool:
    """Return True if `identity` has exceeded `limit` events in the window.

    Uses a Redis fixed-window counter (INCR + EXPIRE NX), falling back to an
    in-process sliding-window bucket when Redis is unavailable.
    """
    key = f"rl:{scope}:" + hashlib.sha256(identity.lower().encode()).hexdigest()
    try:
        client = get_client()
        window = max(window, 1)
        counter_key = f"{key}:{int(time.time() // window)}"
        pipe = client.pipeline()
        pipe.incr(counter_key)
        pipe.expire(counter_key, window, nx=True)
        count, _ = pipe.execute()
        return count > limit
    except Exception as e:
        _log_down("rate_limit", e)
        return _fallback_rate_limit_exceeded(key, limit, window)


def _fallback_rate_limit_exceeded(key: str, limit: int, window: int) -> bool:
    now = time.monotonic()
    dq = _fallback_buckets[key]
    while dq and now - dq[0] > window:
        dq.popleft()
    if len(dq) >= limit:
        return True
    dq.append(now)
    return False


def cache_set(key: str, value: dict | list, ttl: int = 1800):
    try:
        get_client().set(key, json.dumps(value), ex=ttl)
    except Exception as e:
        _log_down("cache_set", e)


def cache_get(key: str) -> dict | list | None:
    try:
        raw = get_client().get(key)
        return json.loads(raw) if raw else None
    except Exception as e:
        _log_down("cache_get", e)
        return None


def publish_signal(signal: dict):
    try:
        get_client().publish("signals", json.dumps(signal))
    except Exception as e:
        _log_down("publish_signal", e)


def add_to_dedup_set(url: str, ttl: int = 86400) -> bool:
    """Returns True if the URL is new (not seen before).

    When Redis is unreachable, returns True so collection proceeds rather than
    silently dropping content as if it had been seen.
    """
    try:
        key = "seen:" + hashlib.sha256(url.encode()).hexdigest()
        result = get_client().set(key, 1, ex=ttl, nx=True)
        return result is True
    except Exception as e:
        _log_down("add_to_dedup_set", e)
        return True
