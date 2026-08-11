import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Prove (or refute) that the LSTM's probability output carries real signal.

For each sentiment-backed ticker it runs the CAUSAL walk-forward LSTM over the
backfilled history and compares it against the no-signal baselines (momentum,
majority prior) with statistical tests:

  * McNemar test (LSTM vs momentum)      — is the accuracy difference real?
  * Wilson binomial CI on LSTM accuracy  — is it distinguishable from 50%?
  * ROC-AUC / balanced accuracy per fold — ranking quality, not just threshold

It then writes per-ticker serving decisions for /api/predictions:

  * buy/sell thresholds fit on the walk-forward OOF ROC (Youden's J), fit on
    ALL folds but the last so the decision rule never sees its own test data,
  * a gate flag: only true when LSTM beats momentum at p < 0.05, which is what
    lets the API emit BUY/SELL. Unproven tickers get NO_SIGNAL at serve time,
    and
  * a last-fold holdout: the fitted thresholds are applied to the final fold and
    reported as holdout_gated_acc / holdout_no_signal_share — the honest gated
    accuracy the API would have produced out-of-sample.

Requires the 5y GDELT sentiment backfill + prices/market_index backfill:

    docker-compose exec api python scripts/backfill_sentiment.py
    docker-compose exec api python scripts/fetch_historical.py --period 5y
    docker-compose exec api python scripts/eval_lstm_signal.py --json-out models/lstm_signal_report.json
"""

from services import lstm_predictor as lsp
from services import walkforward as wf


def _per_ticker(ticker, days, n_folds, horizon, threshold, momentum_window):
    rows = lsp.fetch_live_daily_context(ticker, days=days)
    if len(rows) < 60:
        return None, f"only {len(rows)} rows (< 60) — need the 5y backfill"
    prices = [r["price"] for r in rows]

    oof = wf.evaluate_oof(
        prices,
        n_folds=n_folds,
        horizon=horizon,
        threshold_pct=threshold,
        momentum_window=momentum_window,
        models=["lstm", "momentum", "majority_prior"],
        rows=rows,
    )
    if oof is None:
        return None, "not enough labeled windows for walk-forward folds"

    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    y = np.asarray(oof["lstm"]["true"])
    probs = np.asarray(oof["lstm"]["prob"])
    pred_lstm = np.asarray(oof["lstm"]["pred"])
    pred_mom = np.asarray(oof["momentum"]["pred"])
    pred_maj = np.asarray(oof["majority_prior"]["pred"])

    acc_lstm = float(np.mean(pred_lstm == y))
    acc_mom = float(np.mean(pred_mom == y))
    acc_maj = float(np.mean(pred_maj == y))
    bal = float(balanced_accuracy_score(y, pred_lstm))
    auc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else None
    ci = wf.binomial_ci(len(y), int(np.sum(pred_lstm == y)))
    p_mc = wf.mcnemar_pvalue(y, pred_lstm, pred_mom)
    gate = bool(p_mc < 0.05 and acc_lstm > acc_mom)

    # Youden buy/sell thresholds are fit on all folds but the last (so the
    # decision rule never leaks its own test data), then the last fold is held
    # out to measure the gated decision the API would actually emit.
    fit_folds, holdout = wf.split_holdout(oof, "lstm")
    fit = wf.oof_pooled(oof, "lstm", folds=[f["fold"] for f in fit_folds]) if fit_folds else None
    buy_threshold = lsp.youden_threshold(fit["true"], fit["prob"]) if fit is not None and len(fit["true"]) else 0.5
    sell_threshold = lsp.youden_threshold(1 - fit["true"], fit["prob"]) if fit is not None and len(fit["true"]) else 0.5
    holdout_gated_acc, holdout_no_signal_share = (
        wf.holdout_gated_metrics(
            holdout["prob"], holdout["true"], buy_threshold, sell_threshold, gate)
        if holdout is not None else (None, None)
    )
    threshold_note = (
        f"Youden buy/sell thresholds fit on folds {[f['fold'] for f in fit_folds]} "
        f"(all but last, {int(len(fit['true'])) if fit is not None else 0} windows); "
        f"holdout_gated_acc / holdout_no_signal_share are measured on the held-out "
        f"last fold ({int(len(holdout['true'])) if holdout is not None else 0} windows)."
    )

    report = {
        "ticker": ticker,
        "windows": int(len(y)),
        "up_share": round(float(y.mean()), 4),
        "lstm_acc": round(acc_lstm, 4),
        "lstm_ci95": [round(ci[0], 4), round(ci[1], 4)],
        "momentum_acc": round(acc_mom, 4),
        "majority_acc": round(acc_maj, 4),
        "balanced_accuracy": round(bal, 4),
        "auc": round(auc, 4) if auc is not None else None,
        "p_vs_momentum": round(p_mc, 4),
        "gate": gate,
        "holdout_gated_acc": round(holdout_gated_acc, 4) if holdout_gated_acc is not None else None,
        "holdout_no_signal_share": round(holdout_no_signal_share, 4) if holdout_no_signal_share is not None else None,
    }

    meta = {
        "n_windows": report["windows"],
        "lstm_acc": report["lstm_acc"],
        "momentum_acc": report["momentum_acc"],
        "majority_acc": report["majority_acc"],
        "balanced_accuracy": report["balanced_accuracy"],
        "auc": report["auc"],
        "p_vs_momentum": report["p_vs_momentum"],
        "buy_threshold": round(buy_threshold, 4),
        "sell_threshold": round(sell_threshold, 4),
        "gate": report["gate"],
        "holdout_gated_acc": report["holdout_gated_acc"],
        "holdout_no_signal_share": report["holdout_no_signal_share"],
        "threshold_note": threshold_note,
    }
    return report, meta


def main():
    parser = argparse.ArgumentParser(description="Walk-forward significance + serving thresholds for the LSTM signal")
    parser.add_argument("--tickers", default=",".join(lsp.TRACKED_TICKERS))
    parser.add_argument("--days", type=int, default=1900, help="bars of history to evaluate (~5y)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=lsp.HORIZON)
    parser.add_argument("--threshold", type=float, default=lsp.LABEL_THRESHOLD)
    parser.add_argument("--momentum-window", type=int, default=5)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--thresholds-out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "predict_thresholds.json"))
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    reports, metas, skipped = [], {}, []
    for ticker in tickers:
        print(f"\n=== {ticker} ===")
        report, meta = _per_ticker(
            ticker, args.days, args.n_folds, args.horizon, args.threshold, args.momentum_window)
        if report is None:
            print(f"  skipped: {meta}")
            skipped.append({"ticker": ticker, "reason": meta})
            continue
        reports.append(report)
        metas[ticker] = meta
        print(f"  windows={report['windows']}  up={report['up_share']:.0%}")
        print(f"  lstm={report['lstm_acc']:.1%}  momentum={report['momentum_acc']:.1%}  "
              f"majority={report['majority_acc']:.1%}  bal={report['balanced_accuracy']:.1%}  "
              f"auc={report['auc'] if report['auc'] is not None else 'N/A'}")
        print(f"  LSTM 95% CI: [{report['lstm_ci95'][0]:.1%}, {report['lstm_ci95'][1]:.1%}]")
        print(f"  McNemar p vs momentum: {report['p_vs_momentum']:.4f}  →  gate={'OPEN' if report['gate'] else 'CLOSED'}")
        print(f"  Youden buy>={meta['buy_threshold']:.3f}  sell<={meta['sell_threshold']:.3f}")
        print(f"  holdout: gated_acc={report['holdout_gated_acc'] if report['holdout_gated_acc'] is not None else 'N/A'}  "
              f"no_signal_share={report['holdout_no_signal_share'] if report['holdout_no_signal_share'] is not None else 'N/A'}")

    if args.thresholds_out and metas:
        os.makedirs(os.path.dirname(args.thresholds_out) or ".", exist_ok=True)
        with open(args.thresholds_out, "w") as f:
            json.dump(metas, f, indent=2)
        print(f"\nServing thresholds/gates written → {args.thresholds_out}")

    if args.json_out:
        report_doc = {
            "params": vars(args),
            "per_ticker": reports,
            "skipped": skipped,
            "note": "gate=true requires p_vs_momentum<0.05 AND lstm_acc>momentum_acc; "
                    "API emits BUY/SELL only for gated tickers, else NO_SIGNAL. "
                    "Youden thresholds are fit on all folds but the last; "
                    "holdout_gated_acc/holdout_no_signal_share measure the gated "
                    "decision on the held-out last fold.",
        }
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(report_doc, f, indent=2)
        print(f"Report written → {args.json_out}")

    gated = [r["ticker"] for r in reports if r["gate"]]
    print(f"\nGated (demonstrated signal): {gated if gated else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
