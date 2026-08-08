import os
import random
import subprocess
import sys

from temporalio import activity


@activity.defn
async def free_collect_activity() -> None:
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Running free collector (Google News + Yahoo Finance)...")
    subprocess.run([sys.executable, os.path.join("scripts", "free_collect.py")], cwd=backend_dir, check=True)
    print("Free collection complete.")


@activity.defn
async def seed_demo_trades_activity() -> None:
    from services import influx
    from tickers import TICKERS

    tickers = TICKERS
    prices = {"AAPL": 283.78, "TSLA": 379.71, "NVDA": 192.53, "MSFT": 372.97, "GOOGL": 337.39, "AMZN": 232.69, "META": 550.25}
    total = 0
    for ticker in tickers:
        price = prices.get(ticker)
        if not price:
            continue
        for _ in range(random.randint(2, 5)):
            side = random.choice(["buy", "sell"])
            price_pt = price * (1 + random.uniform(-0.005, 0.005))
            qty = random.randint(5, 100)
            influx.write_trade(ticker, side, round(price_pt, 2), qty, is_demo=True)
            total += 1
    print(f"Demo trades written ({total} total).")
