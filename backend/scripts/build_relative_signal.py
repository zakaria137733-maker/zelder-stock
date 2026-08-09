"""Walk-forward experiment: can a model predict relative outperformance vs SPY?

Target per ticker per day: y = 1 if the ticker's return over the next 5 trading
days beats SPY's return over the same window. Base rate is ~50% by construction,
so "real signal" means beating 50% AND beating a point-in-time momentum baseline
(ticker 20d momentum > SPY 20d momentum) on out-of-sample days.

Model: sklearn HistGradientBoostingClassifier on engineered daily features.
Evaluation: expanding-window walk-forward, retrained every STEP days, metrics
pooled over out-of-sample days only, with per-day paired comparison against the
momentum baseline (Wilcoxon) to avoid over-crediting correlated same-day rows.

    docker-compose exec api python scripts/build_relative_signal.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.lstm_predictor import fetch_training_data

INITIAL_TRAIN_DATE = "2023-08-01"
STEP_DAYS = 21
HORIZON = 5
MOM_BASELINE_WINDOW = 20

FEATURES = [
        "r5", "r10", "r20", "r60", "r120",
        "rel_r5", "rel_r10", "rel_r20", "rel_r60",
    "ma20_dist", "ma50_dist", "ma200_dist",
    "rsi", "adx", "cci", "stoch", "bb_width",
    "atr_close", "vol20", "vol_momentum",
    "spy_mom20", "vix", "vix_chg20",
    "sentiment", "sent_chg5",
    "rank_rel10", "rank_rel20", "rank_r10",
]

MODEL_KWARGS = dict(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42,
)


def build_panel(data: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(data)
    df = df.dropna(subset=["price"]).copy()
    df["date"] = pd.to_datetime(df["time"]).dt.date
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    ticker_groups = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        p = g["price"]
        r1 = p.pct_change()
        spy_close = (1.0 + g["spy_ret"] / 100.0).cumprod() * 100.0
        spy_r5 = spy_close.pct_change(5)

        g["r5"] = p.pct_change(5)
        g["r10"] = p.pct_change(10)
        g["r20"] = p.pct_change(20)
        g["r60"] = p.pct_change(60)
        g["r120"] = p.pct_change(120)
        g["rel_r5"] = g["r5"] - spy_r5
        g["rel_r10"] = g["r10"] - spy_close.pct_change(10)
        g["rel_r20"] = g["r20"] - spy_close.pct_change(20)
        g["rel_r60"] = g["r60"] - spy_close.pct_change(60)

        g["ma20_dist"] = p / p.rolling(20).mean() - 1.0
        g["ma50_dist"] = p / p.rolling(50).mean() - 1.0
        g["ma200_dist"] = p / p.rolling(200).mean() - 1.0

        g["atr_close"] = g["atr"] / p.replace(0, np.nan)
        g["vol20"] = r1.rolling(20).std()
        g["spy_mom20"] = spy_close.pct_change(20)
        g["vix_chg20"] = g["vix"] - g["vix"].rolling(20).mean()
        g["sent_chg5"] = g["sentiment"] - g["sentiment"].rolling(5).mean()

        fwd_ticker = p.pct_change(HORIZON).shift(-HORIZON)
        fwd_spy = spy_close.pct_change(HORIZON).shift(-HORIZON)
        g["y"] = (fwd_ticker > fwd_spy).astype(float)
        g["rel_fwd"] = (fwd_ticker - fwd_spy) * 100.0
        g["mom_pred20"] = (g["rel_r20"] > 0).astype(float)
        g["mom_pred_same"] = (g[f"rel_r{HORIZON}"] > 0).astype(float)

        ticker_groups.append(g)

    panel = pd.concat(ticker_groups, ignore_index=True)

    panel["rank_rel10"] = panel.groupby("date")["rel_r10"].rank(pct=True)
    panel["rank_rel20"] = panel.groupby("date")["rel_r20"].rank(pct=True)
    panel["rank_r10"] = panel.groupby("date")["r10"].rank(pct=True)
    return panel.dropna(subset=FEATURES + ["y"])


def walk_forward(panel: pd.DataFrame, init_date: str, step: int) -> dict:
    panel = panel.sort_values("date").reset_index(drop=True)
    test_dates = sorted(panel["date"].unique())
    split_idx = 0
    for i, d in enumerate(test_dates):
        if str(d) >= init_date:
            split_idx = i
            break
    test_start_dates = test_dates[split_idx::step]

    oos = []
    for start in test_start_dates:
        end = start + pd.Timedelta(days=step * 2)
        train = panel[panel["date"] < start]
        test = panel[(panel["date"] >= start) & (panel["date"] < end)]
        if len(train) < 200 or len(test) == 0:
            continue
        model = HistGradientBoostingClassifier(**MODEL_KWARGS)
        model.fit(train[FEATURES], train["y"])
        proba = model.predict_proba(test[FEATURES])[:, 1]
        oos.append(test.assign(prob_up=proba, pred=(proba > 0.5).astype(float)))

    return pd.concat(oos, ignore_index=True)


def summarize(oos: pd.DataFrame) -> dict:
    y = oos["y"].values
    pred = oos["pred"].values
    prob = oos["prob_up"].values
    base_rate = float(y.mean())

    out = {
        "n_samples": int(len(y)),
        "n_days": int(oos["date"].nunique()),
        "base_rate": round(base_rate, 4),
        "majority_acc": round(max(base_rate, 1 - base_rate), 4),
        "model_acc": round(float((pred == y).mean()), 4),
        "mom20_acc": round(float((oos["mom_pred20"] == y).mean()), 4),
        "mom_same_acc": round(float((oos["mom_pred_same"] == y).mean()), 4),
        "auc": round(float(roc_auc_score(y, prob)), 4),
        "brier": round(float(brier_score_loss(y, prob)), 4),
    }

    day = oos.groupby("date").agg(
        day_model=("pred", lambda s: float((s == oos.loc[s.index, "y"]).mean())),
        day_mom20=("mom_pred20", lambda s: float((s == oos.loc[s.index, "y"]).mean())),
        day_mom_same=("mom_pred_same", lambda s: float((s == oos.loc[s.index, "y"]).mean())),
        day_base=("y", "mean"),
    )
    out["day_mean_model_acc"] = round(float(day["day_model"].mean()), 4)
    out["day_mean_mom20_acc"] = round(float(day["day_mom20"].mean()), 4)
    out["day_mean_mom_same_acc"] = round(float(day["day_mom_same"].mean()), 4)
    out["day_mean_base_rate"] = round(float(day["day_base"].mean()), 4)

    if len(day) >= 8:
        _, p_model_vs_50 = stats.wilcoxon(day["day_model"] - 0.5)
        _, p_model_vs_mom20 = stats.wilcoxon(day["day_model"] - day["day_mom20"])
        _, p_model_vs_mom_same = stats.wilcoxon(day["day_model"] - day["day_mom_same"])
        out["wilcoxon_model_vs_50_p"] = float(p_model_vs_50)
        out["wilcoxon_model_vs_mom20_p"] = float(p_model_vs_mom20)
        out["wilcoxon_model_vs_mom_same_p"] = float(p_model_vs_mom_same)

    per_ticker = {}
    for t, g in oos.groupby("ticker"):
        per_ticker[t] = {
            "n": int(len(g)),
            "model_acc": round(float((g["pred"] == g["y"]).mean()), 4),
            "mom20_acc": round(float((g["mom_pred20"] == g["y"]).mean()), 4),
            "mom_same_acc": round(float((g["mom_pred_same"] == g["y"]).mean()), 4),
        }
    out["per_ticker"] = per_ticker
    out["trade_stats"] = trade_stats(oos)
    return out


def trade_stats(oos: pd.DataFrame, thresholds=(0.52, 0.55, 0.58, 0.60, 0.65)) -> dict:
    out = {}
    for th in thresholds:
        sel = oos[oos["prob_up"] > th]
        if len(sel) == 0:
            out[str(th)] = {"n_trades": 0, "trade_freq": 0.0}
            continue
        win = float((sel["y"] == 1).mean())
        rel = float(sel["rel_fwd"].mean())
        se = float(sel["rel_fwd"].std() / np.sqrt(len(sel)))
        out[str(th)] = {
            "n_trades": int(len(sel)),
            "trade_freq": round(len(sel) / len(oos), 3),
            "win_rate": round(win, 3),
            "avg_rel_ret": round(rel, 3),
            "tstat": round(rel / se, 2) if se else None,
        }
    return out


def print_report(r: dict):
    print("\n=== RELATIVE SIGNAL WALK-FORWARD REPORT ===")
    print(f"samples: {r['n_samples']} ({r['n_days']} days, 7 tickers pooled)")
    print(f"base rate (ticker beats SPY over {HORIZON}d): {r['base_rate']:.3f}  |  majority acc: {r['majority_acc']:.3f}")
    print(f"MODEL OOS accuracy:   {r['model_acc']:.3f}")
    print(f"momentum(20d) acc:    {r['mom20_acc']:.3f}")
    print(f"momentum({HORIZON}d) acc:  {r['mom_same_acc']:.3f}")
    print(f"AUC: {r['auc']:.3f}  |  Brier: {r['brier']:.4f}")
    print(f"per-day model acc: {r['day_mean_model_acc']:.3f} vs mom20 {r['day_mean_mom20_acc']:.3f} vs mom{HORIZON} {r['day_mean_mom_same_acc']:.3f} vs base {r['day_mean_base_rate']:.3f}")
    for k in ("wilcoxon_model_vs_50_p", "wilcoxon_model_vs_mom20_p", "wilcoxon_model_vs_mom_same_p"):
        if k in r:
            print(f"{k}: {r[k]:.4f}")
    print("per-ticker:")
    for t, m in r["per_ticker"].items():
        print(f"  {t}: n={m['n']} model={m['model_acc']:.3f} mom20={m['mom20_acc']:.3f} mom{HORIZON}={m['mom_same_acc']:.3f}")
    print("trade-level (prob_up threshold -> win rate / avg rel ret vs SPY over horizon):")
    for th, s in r["trade_stats"].items():
        if s["n_trades"] == 0:
            print(f"  prob>{th}: no trades")
            continue
        print(f"  prob>{th}: n={s['n_trades']} freq={s['trade_freq']:.2f} win={s['win_rate']:.3f} "
              f"avg_rel_ret={s['avg_rel_ret']:+.2f}pp t={s['tstat']}")


def main():
    global HORIZON, FEATURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-date", default=INITIAL_TRAIN_DATE)
    parser.add_argument("--step", type=int, default=STEP_DAYS)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--no-sentiment", action="store_true",
                        help="drop sentiment features entirely")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    HORIZON = args.horizon
    if args.no_sentiment:
        FEATURES = [f for f in FEATURES if f not in ("sentiment", "sent_chg5")]

    print("Fetching training data...")
    data = fetch_training_data()
    panel = build_panel(data)
    print(f"panel rows (labeled, feature-complete): {len(panel)}")

    oos = walk_forward(panel, args.init_date, args.step)
    print(f"out-of-sample rows: {len(oos)}")
    report = summarize(oos)
    print_report(report)

    if args.json_out:
        import json
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"report written to {args.json_out}")


if __name__ == "__main__":
    main()
