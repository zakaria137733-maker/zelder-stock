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
        prices = [r["price"] for r in rows]
        return prices, f"{args.ticker} - InfluxDB ({len(rows)} daily bars)", rows
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
        rows = data.get("rows") if isinstance(data, dict) else None
        if rows is not None and len(rows) != len(prices):
            print("rows and prices must be the same length; ignoring rows.")
            rows = None
        return prices, f"{args.input} ({len(prices)} bars)", rows
    prices = wf.synthetic_prices(seed=args.seed)
    return prices, f"synthetic random walk - seed={args.seed} (offline)", None


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward eval vs three no-signal baselines")
    parser.add_argument("--ticker", default="", help="evaluate real daily data for a ticker (InfluxDB)")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--input", default="", help="JSON file with a price list or {prices: [...]}")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--momentum-window", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="label move threshold in %% (ignored with --thresholds)")
    parser.add_argument("--thresholds", default="",
                        help="comma list, e.g. '0.5,1.0,1.5,2.0' to sweep the label threshold")
    parser.add_argument("--models", default="",
                        help="comma list to restrict methods, e.g. 'logistic,xgboost,lstm' "
                             "(lstm is only available with --ticker/--input rows)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    prices, source, rows = load_prices(args)
    models = [m.strip() for m in args.models.split(",") if m.strip()] or None
    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()] or [args.threshold]

    rows_out = []
    for thresh in thresholds:
        result = wf.evaluate(
            prices,
            n_folds=args.n_folds,
            horizon=args.horizon,
            threshold_pct=thresh,
            momentum_window=args.momentum_window,
            models=models,
            rows=rows,
        )
        if result is None:
            print(f"Not enough labelled windows at threshold {thresh:g}%.")
            continue
        rows_out.append((thresh, result))

    if not rows_out:
        return 1

    print(f"Source: {source}")
    if len(rows_out) == 1:
        thresh, result = rows_out[0]
        print(f"Label: {args.horizon}-day forward move +/-{thresh:g}% - windows: {result['n_windows_total']} - "
              f"folds: {result['n_folds_run']}\n")
        print(f"{'method':<16}{'fold accs':<42}{'mean':>7}")
        print("-" * 65)
        for name, mean_acc in result["overall"].items():
            fold_accs = "  ".join(f"{r[name]:.1%}" for r in result["folds"])
            print(f"{name:<16}{fold_accs:<42}{mean_acc:>7.1%}")
        print(f"{'coin flip (ref)':<16}{'':<42}{0.5:>7.1%}")
    else:
        names = list(rows_out[0][1]["overall"].keys())
        print(f"Label: {args.horizon}-day forward move beyond threshold - windows/fold vary by threshold\n")
        print(f"{'threshold':<12}{'windows':>8}" + "".join(f"{n:>12}" for n in names) + f"{'coin(0.5)':>12}")
        print("-" * (12 + 8 + 12 * len(names) + 12))
        for thresh, result in rows_out:
            line = f"{thresh:g}%".ljust(12) + f"{result['n_windows_total']:>8}"
            for name in names:
                line += f"{result['overall'][name]:>12.1%}"
            line += f"{0.5:>12.1%}"
            print(line)

    print("\nWhat counts as signal: any method clearly above the majority/momentum baselines.")
    if args.json_out:
        report = {
            "source": source,
            "params": vars(args),
            "results": [
                {"threshold_pct": thresh, **result} for thresh, result in rows_out
            ],
        }
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
