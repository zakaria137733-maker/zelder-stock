import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Score the DEPLOYED model artifact (the committed lstm_model.pt + scaler.json) on
recent held-out daily data, using the exact serving pipeline behind /api/predictions.

This is the honest number to quote: it measures the model that is actually shipped
and served, not a freshly retrained one.

Usage:
    python scripts/eval_deployed.py AAPL --days 120 --horizon 5
    python scripts/eval_deployed.py AAPL --days 120 --json-out models/eval_report.json
"""

from services import lstm_predictor as lsp


def run(ticker: str, days: int, horizon: int, json_out: str) -> int:
    print(f"Fetching {days} days of daily context for {ticker}...")
    rows = lsp.fetch_live_daily_context(ticker, days=days)
    if len(rows) < lsp.SEQUENCE_LEN + horizon:
        print(f"Not enough data: only {len(rows)} daily rows for {ticker}")
        return 1

    X, y = lsp.build_eval_sequences(rows, horizon=horizon)
    if len(X) == 0:
        print("No labeled windows — prices did not move ±1% anywhere in the window.")
        return 1

    print(f"Evaluated windows: {len(X)} ({int(y.sum())} up / {int(len(y) - y.sum())} down)")
    result = lsp.evaluate_deployed(X, y)
    if result is None:
        print("No deployed model artifact found (MODEL_DIR=" + lsp.MODEL_DIR + ").")
        return 1

    acc, probs = result
    preds = (probs > 0.5).astype(float)
    up_acc = float((preds[y == 1] == 1).mean()) if (y == 1).any() else None
    down_acc = float((preds[y == 0] == 0).mean()) if (y == 0).any() else None

    report = {
        "ticker": ticker,
        "days": days,
        "horizon": horizon,
        "windows": int(len(y)),
        "up_samples": int(y.sum()),
        "down_samples": int(len(y) - y.sum()),
        "accuracy": round(acc, 4),
        "accuracy_up": round(up_acc, 4) if up_acc is not None else None,
        "accuracy_down": round(down_acc, 4) if down_acc is not None else None,
        "baseline": round(max(y.mean(), 1 - y.mean()), 4),
    }

    print("\n" + "=" * 46)
    print("DEPLOYED MODEL EVALUATION")
    print("=" * 46)
    for key, value in report.items():
        print(f"  {key:14} {value}")
    print("=" * 46)

    if json_out:
        os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {json_out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the deployed model artifact on recent data")
    parser.add_argument("ticker", nargs="?", default="AAPL")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    raise SystemExit(run(args.ticker, args.days, args.horizon, args.json_out))
