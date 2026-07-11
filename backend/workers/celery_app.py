import asyncio
import subprocess
import sys
from celery import Celery
from celery.schedules import crontab
from config import settings

celery_app = Celery(
    "sentimentiq",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    beat_schedule={
        "collect-free-every-5min": {
            "task": "workers.celery_app.free_collect_task",
            "schedule": crontab(minute="*/5"),
        },
        "seed-demo-trades-every-5min": {
            "task": "workers.celery_app.seed_demo_trades_task",
            "schedule": crontab(minute="*/5"),
        },
    },
)


@celery_app.task(name="workers.celery_app.free_collect_task")
def free_collect_task():
    print("Running free collector (Google News + Yahoo Finance)...")
    subprocess.run([sys.executable, "free_collect.py"], cwd="/app")
    print("Free collection complete.")


@celery_app.task(name="workers.celery_app.seed_demo_trades_task")
def seed_demo_trades_task():
    import random
    from services import influx
    tickers = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]
    prices = {"AAPL": 283.78, "TSLA": 379.71, "NVDA": 192.53, "MSFT": 372.97, "GOOGL": 337.39, "AMZN": 232.69, "META": 550.25}
    for _ in range(random.randint(2, 5)):
        ticker = random.choice(tickers)
        side = random.choice(["buy", "sell"])
        price = prices[ticker] * (1 + random.uniform(-0.005, 0.005))
        qty = random.randint(5, 100)
        influx.write_trade(ticker, side, round(price, 2), qty)
    print("Demo trades written.")
