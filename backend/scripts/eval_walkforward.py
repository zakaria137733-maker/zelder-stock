import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Time-ordered walk-forward evaluation against three "no signal" baselines:
coin flip (0.5), majority-up prior, and trailing momentum continuation.

Fully offline by default (synthetic data). Point it at real data with:

    python scripts/eval_walkforward.py --ticker AAPL --days 365
    python scripts/eval_walkforward.py --input prices.json
"""

from services import walkforward as wf


def load_prices(args):
    if args.ticker:
        from services import lstm_predictor as lsp

        rows = lsp.fetch_live_daily_context(args.ticker, days=args.days)
        if len(rows) < 30:
            print(f"Not enough daily rows for {args.ticker} ({len(rows)}).")
            sys.exit(1)
        return [r["price"] for r in rows], f"{args.ticker} - InfluxDB ({len(rows)} daily bars)"
    if args.input:
        with open(args.input) as f:
            data = json.load(f)
        if isinstance(data, list):
            prices = data
        elif isinstance(data, dict):
            prices = data.get("prices") or [r["price"] for r in data.get("rows", [])]
        if not prices or len(prices) < 30:
            print(f"No usable prices in {args.input}.")
            sys.exit(1)
        return prices, f"{args.input} ({len(prices)} bars)"
    prices = wf.synthetic_prices(seed=args.seed)
    return prices, f"synthetic random walk - seed={args.seed} (offline)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward eval vs three no-signal baselines")
    parser.add_argument("--ticker", default="", help="evaluate real daily data for a ticker (InfluxDB)")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--input", default="", help="JSON file with a price list or {prices: [...]}")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--momentum-window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    prices, source = load_prices(args)
    result = wf.evaluate(
        prices,
        n_folds=args.n_folds,
        horizon=args.horizon,
        momentum_window=args.momentum_window,
    )
    if result is None:
        print("Not enough labelled windows to evaluate.")
        return 1

    print(f"Source: {source}")
    print(f"Label: {args.horizon}-day forward move +/-1% - windows: {result['n_windows_total']} - "
          f"folds: {result['n_folds_run']}\n")
    print(f"{'method':<16}{'fold accs':<42}{'mean':>7}")
    print("-" * 65)
    rows = result["folds"]
    for name, mean_acc in result["overall"].items():
        fold_accs = "  ".join(f"{r[name]:.1%}" for r in rows)
        print(f"{name:<16}{fold_accs:<42}{mean_acc:>7.1%}")
    print(f"{'coin flip (ref)':<16}{'':<42}{0.5:>7.1%}")

    print("\nWhat counts as signal: any method clearly above the majority/momentum baselines.")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"source": source, "params": vars(args), **result}, f, indent=2)
        print(f"Report written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
