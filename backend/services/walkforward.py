"""Time-ordered walk-forward evaluation harness.

Every method is evaluated *causally*: a prediction for a test window may only use
data up to that window's final bar. Results are reported against three baselines
that define "no real signal":

  * coin flip      — 0.5 expected accuracy (reference line)
  * majority prior — always predict the label most common in the training folds
  * momentum       — predict continuation of the trailing N-day return

A simple causal logistic model is included so the harness can be exercised
end-to-end without touching the LSTM. The label convention matches the model
training code (build_sequences): forward return over `horizon` bars, > +1% up,
< -1% down, neutral skipped.

Driver: `python scripts/eval_walkforward.py --help`
"""

from __future__ import annotations

import numpy as np


def trailing_return(prices, i, window):
    """Percent return of prices[i] vs prices[i - window]. Returns 0.0 causally when
    there is no prior bar (i < window) — the same convention as the feature builder."""
    if i - window < 0:
        return 0.0
    base = prices[i - window]
    if not base:
        return 0.0
    return (prices[i] - base) / base * 100.0


def binary_labels(prices, horizon=5, threshold_pct=1.0, start=10):
    """Time-ordered (i, label) windows with a forward move beyond ±threshold_pct."""
    labels = []
    for i in range(start, len(prices) - horizon):
        base = prices[i]
        pct = (prices[i + horizon] - base) / base * 100.0 if base else 0.0
        if pct > threshold_pct:
            labels.append((i, 1))
        elif pct < -threshold_pct:
            labels.append((i, 0))
    return labels


def synthetic_prices(n=600, seed=42, momentum=0.12, drift=0.0001, vol=0.02):
    """Geometric random walk with weak return autocorrelation (realistic-ish daily data)."""
    rng = np.random.default_rng(seed)
    rets = rng.standard_normal(n) * vol + drift
    for i in range(1, n):
        rets[i] = rets[i] + momentum * rets[i - 1]
    prices = [100.0]
    for r in rets:
        prices.append(prices[-1] * (1 + r))
    return prices


def _majority_prior(prices, train_idxs, train_ys, test_idxs):
    majority = 1 if train_ys.mean() >= 0.5 else 0
    return np.full(len(test_idxs), majority)


def _momentum(window):
    def predict(prices, train_idxs, train_ys, test_idxs):
        return np.array([1 if trailing_return(prices, int(i), window) >= 0 else 0 for i in test_idxs])

    return predict


def _trailing_features(prices, idxs, lookback=(1, 5, 10)):
    """Shared causal feature rows: trailing N-day returns + trailing 5-day vol."""
    rows = []
    for i in idxs:
        i = int(i)
        row = [trailing_return(prices, i, w) for w in lookback]
        if i >= 5:
            daily = []
            for j in range(1, 6):
                prev = prices[i - j]
                if prev:
                    daily.append((prices[i - j + 1] - prev) / prev * 100.0)
            row.append(float(np.std(daily)) if daily else 0.0)
        else:
            row.append(0.0)
        rows.append(row)
    return np.array(rows)


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def _logistic(prices, train_idxs, train_ys, test_idxs, lookback=(1, 5, 10)):
    from sklearn.linear_model import LogisticRegression

    if len(train_ys) < 10 or len(np.unique(train_ys)) < 2:
        return np.full(len(test_idxs), float(train_ys.mean() >= 0.5))
    clf = LogisticRegression(max_iter=2000)
    clf.fit(_trailing_features(prices, train_idxs, lookback), np.array(train_ys, dtype=float))
    return clf.predict(_trailing_features(prices, test_idxs, lookback))


def _xgboost(prices, train_idxs, train_ys, test_idxs, lookback=(1, 5, 10)):
    from xgboost import XGBClassifier

    if len(train_ys) < 20 or len(np.unique(train_ys)) < 2:
        return np.full(len(test_idxs), float(train_ys.mean() >= 0.5))
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=1,
        eval_metric="logloss",
    )
    clf.fit(_trailing_features(prices, train_idxs, lookback), np.array(train_ys, dtype=int))
    return clf.predict(_trailing_features(prices, test_idxs, lookback))


def _mlp(prices, train_idxs, train_ys, test_idxs, lookback=(1, 5, 10)):
    from sklearn.neural_network import MLPClassifier

    if len(train_ys) < 20 or len(np.unique(train_ys)) < 2:
        return np.full(len(test_idxs), float(train_ys.mean() >= 0.5))
    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
    )
    clf.fit(_trailing_features(prices, train_idxs, lookback), np.array(train_ys, dtype=int))
    return clf.predict(_trailing_features(prices, test_idxs, lookback))


def evaluate(
    prices,
    n_folds=5,
    horizon=5,
    threshold_pct=1.0,
    start=10,
    momentum_window=5,
    min_train_windows=20,
    models=None,
):
    """Expanding-window walk-forward evaluation.

    Folds are contiguous time slices over the binary-labelled windows. Fold 0 is
    only used for training; folds 1..n_folds-1 are each tested once, training on
    all windows before the fold. `models` optionally restricts the methods
    (default: all available). Returns a summary dict, or None if there is not
    enough data.
    """
    labels = binary_labels(prices, horizon, threshold_pct, start)
    n = len(labels)
    fold_size = n // n_folds
    if n < min_train_windows + fold_size or fold_size == 0:
        return None

    methods = {
        "majority_prior": _majority_prior,
        "momentum": _momentum(window=momentum_window),
        "logistic": _logistic,
    }
    if _xgboost_available():
        methods["xgboost"] = _xgboost
    methods["mlp"] = _mlp
    if models:
        methods = {k: v for k, v in methods.items() if k in models}

    folds = []
    per_method = {name: [] for name in methods}

    for k in range(1, n_folds):
        start_idx = k * fold_size
        end_idx = n if k == n_folds - 1 else (k + 1) * fold_size
        train = labels[:start_idx]
        test = labels[start_idx:end_idx]
        if len(train) < min_train_windows or not test:
            continue

        train_idxs = np.array([i for i, _ in train], dtype=int)
        train_ys = np.array([y for _, y in train], dtype=int)
        test_idxs = np.array([i for i, _ in test], dtype=int)
        y_true = np.array([y for _, y in test], dtype=int)

        fold = {"fold": k, "windows": int(len(test)), "up_share": float(y_true.mean())}
        for name, fn in methods.items():
            y_pred = np.asarray(fn(prices, train_idxs, train_ys, test_idxs))
            acc = float(np.mean(y_pred == y_true))
            fold[name] = acc
            per_method[name].append(acc)
        folds.append(fold)

    if not folds:
        return None

    return {
        "n_windows_total": n,
        "n_folds_run": len(folds),
        "coin_flip_expected": 0.5,
        "overall": {name: float(np.mean(accs)) for name, accs in per_method.items()},
        "folds": folds,
    }
