import os
import random
import subprocess
import sys

from temporalio import activity


@activity.defn
async def free_collect_activity() -> None:
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Running free collector (Google News + Yahoo Finance)...")
    subprocess.run([sys.executable, "free_collect.py"], cwd=backend_dir, check=True)
    print("Free collection complete.")


@activity.defn
async def seed_demo_trades_activity() -> None:
    from services import influx

    tickers = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]
    prices = {"AAPL": 283.78, "TSLA": 379.71, "NVDA": 192.53, "MSFT": 372.97, "GOOGL": 337.39, "AMZN": 232.69, "META": 550.25}
    total = 0
    for ticker in tickers:
        for _ in range(random.randint(2, 5)):
            side = random.choice(["buy", "sell"])
            price = prices[ticker] * (1 + random.uniform(-0.005, 0.005))
            qty = random.randint(5, 100)
            influx.write_trade(ticker, side, round(price, 2), qty)
            total += 1
    print(f"Demo trades written ({total} total).")
