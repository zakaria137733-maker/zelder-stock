from datetime import UTC, datetime

from services import influx
from services.redis_client import cache_get, cache_set
from tickers import TICKERS

SHIFT_THRESHOLD=10.0
WINDOW_HOURS=3


def detect_alerts()->list[dict]:

    tickers = TICKERS
    alerts = []

    for ticker in tickers:
        history = influx.query_sentiment_history(ticker, hours=WINDOW_HOURS+1)
        if len(history) < 2:
            continue

        current = history[-1]["value"]
        earlier = history[0]["value"]
        shift = current - earlier

        if abs(shift)>=SHIFT_THRESHOLD:
            direction="up" if shift>0 else "down"
            severity = "high" if abs(shift)>=20 else "medium"
            alerts.append({
                "ticker": ticker,
                "current_score": round(current,1),
                "previous_score": round(earlier,1),
                "shift": round(shift,1),
                "direction":direction,
                "severity":severity,
                "window_hours":WINDOW_HOURS,
                "message":f"{ticker} sentiment {direction} {abs(shift):.1f} pts in {WINDOW_HOURS}h",
                "triggered_at":datetime.now(UTC).isoformat(),
            })


    cache_set("alerts:active",alerts,ttl=300)
    return alerts


def get_cached_alerts()->list[dict]:
    cached = cache_get("alerts:active")
    return cached if cached else detect_alerts()
