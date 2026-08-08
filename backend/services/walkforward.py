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


def _lstm_factory(rows, epochs=60, patience=8, seed=42):
    """Causal LSTM method for the walk-forward harness.

    `rows` must be the same length as `prices` (index-aligned daily dicts with
    price/volume/time/sentiment/spy_ret/vix — see fetch_live_daily_context).
    Each fold trains a fresh SentimentLSTM on the training windows (which only
    see data up to their own final bar) and predicts the test windows. Uses the
    same feature/label conventions as services.lstm_predictor, so the resulting
    number is directly comparable to the other methods in this harness.
    """
    def _lstm(prices, train_idxs, train_ys, test_idxs):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        from services.lstm_predictor import SentimentLSTM, build_raw_features

        def build(idxs, ys):
            Xw, yw = [], []
            for i, yv in zip(idxs, ys, strict=True):
                i = int(i)
                w = build_raw_features(rows[: i + 1])
                if w is None:
                    continue
                Xw.append(w)
                yw.append(yv)
            if not Xw:
                return None, None
            return np.array(Xw, dtype=np.float32), np.array(yw, dtype=float)

        X_tr, y_tr = build(train_idxs, train_ys)
        if X_tr is None or len(y_tr) < 20 or len(np.unique(y_tr)) < 2:
            return np.full(len(test_idxs), float(np.mean(train_ys) >= 0.5))

        torch.manual_seed(seed)
        np.random.seed(seed)

        split = int(0.8 * len(y_tr))
        X_tt = torch.tensor(np.clip(X_tr[:split], -5, 5), dtype=torch.float32)
        y_tt = torch.tensor(y_tr[:split], dtype=torch.float32)
        X_tv = torch.tensor(np.clip(X_tr[split:], -5, 5), dtype=torch.float32)
        y_tv = torch.tensor(y_tr[split:], dtype=torch.float32)

        loader = DataLoader(TensorDataset(X_tt, y_tt), batch_size=64, shuffle=True)
        model = SentimentLSTM()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCELoss()

        best_acc = 0
        best_state = None
        no_improve = 0
        for _epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb).view(-1), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                acc = float(((model(X_tv).view(-1) > 0.5).float() == y_tv).float().mean())
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        if best_state:
            model.load_state_dict(best_state)

        X_te, _ = build(test_idxs, np.zeros(len(test_idxs)))
        if X_te is None:
            return np.full(len(test_idxs), float(np.mean(train_ys) >= 0.5))
        model.eval()
        with torch.no_grad():
            probs = model(torch.tensor(np.clip(X_te, -5, 5), dtype=torch.float32)).view(-1).numpy()
        return probs

    return _lstm


def _build_methods(momentum_window, models, rows):
    """Resolve the methods dict for the given config."""
    methods = {
        "majority_prior": _majority_prior,
        "momentum": _momentum(window=momentum_window),
        "logistic": _logistic,
    }
    if _xgboost_available():
        methods["xgboost"] = _xgboost
    methods["mlp"] = _mlp
    if rows is not None:
        methods["lstm"] = _lstm_factory(rows)
    if models:
        methods = {k: v for k, v in methods.items() if k in models}
    return methods


def _iter_folds(labels, n_folds, min_train_windows):
    """Yield (fold_k, train_idxs, train_ys, test_idxs, y_true) in time order."""
    n = len(labels)
    fold_size = n // n_folds
    if fold_size == 0:
        return
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
        yield k, train_idxs, train_ys, test_idxs, y_true


def evaluate(
    prices,
    n_folds=5,
    horizon=5,
    threshold_pct=1.0,
    start=10,
    momentum_window=5,
    min_train_windows=20,
    models=None,
    rows=None,
):
    """Expanding-window walk-forward evaluation.

    Folds are contiguous time slices over the binary-labelled windows. Fold 0 is
    only used for training; folds 1..n_folds-1 are each tested once, training on
    all windows before the fold. `models` optionally restricts the methods
    (default: all available). `rows` optionally provides the daily feature rows
    (index-aligned with `prices`) so the LSTM method is trained/evaluated in the
    same causal framework. Returns a summary dict, or None if there is not
    enough data.
    """
    labels = binary_labels(prices, horizon, threshold_pct, start)
    n = len(labels)
    fold_size = n // n_folds
    if n < min_train_windows + fold_size or fold_size == 0:
        return None

    methods = _build_methods(momentum_window, models, rows)
    folds = []
    per_method = {name: [] for name in methods}

    for k, train_idxs, train_ys, test_idxs, y_true in _iter_folds(labels, n_folds, min_train_windows):
        fold = {"fold": k, "windows": int(len(test_idxs)), "up_share": float(y_true.mean())}
        for name, fn in methods.items():
            raw = np.asarray(fn(prices, train_idxs, train_ys, test_idxs))
            y_pred = (raw > 0.5).astype(float) if name == "lstm" else raw
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


def evaluate_oof(
    prices,
    n_folds=5,
    horizon=5,
    threshold_pct=1.0,
    start=10,
    momentum_window=5,
    min_train_windows=20,
    models=None,
    rows=None,
):
    """Like evaluate() but returns pooled out-of-fold predictions per method.

    Result: ``{method: {"pred": [...], "prob": [...], "true": [...]}}`` in time
    order across all test folds. ``prob`` is populated for methods that emit
    probabilities (currently ``lstm``), enabling ROC thresholds and calibration.
    Returns None when there is not enough data.
    """
    labels = binary_labels(prices, horizon, threshold_pct, start)
    n = len(labels)
    fold_size = n // n_folds
    if n < min_train_windows + fold_size or fold_size == 0:
        return None

    methods = _build_methods(momentum_window, models, rows)
    out = {name: {"pred": [], "prob": [], "true": []} for name in methods}

    for _k, train_idxs, train_ys, test_idxs, y_true in _iter_folds(labels, n_folds, min_train_windows):
        for name, fn in methods.items():
            raw = np.asarray(fn(prices, train_idxs, train_ys, test_idxs))
            if name == "lstm":
                out[name]["prob"].extend(raw.tolist())
                out[name]["pred"].extend((raw > 0.5).astype(float).tolist())
            else:
                out[name]["pred"].extend(raw.tolist())
            out[name]["true"].extend(y_true.tolist())
    return out


def binomial_ci(n, k, alpha=0.05):
    """Wilson score interval for an observed rate k/n (e.g. accuracy)."""
    if n == 0:
        return (0.0, 0.0)
    from scipy.stats import norm

    p = k / n
    z = norm.ppf(1 - alpha / 2.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def mcnemar_pvalue(y_true, y_a, y_b):
    """Two-sided exact McNemar test that two classifiers disagree asymmetrically.

    Tests whether classifier A is significantly more (or less) accurate than B,
    using only the discordant pairs. A p < 0.05 means the accuracy difference is
    unlikely under the null of equal error rates.
    """
    from scipy.stats import binomtest

    y_true = np.asarray(y_true)
    y_a = np.asarray(y_a)
    y_b = np.asarray(y_b)
    d01 = int(np.sum((y_a != y_true) & (y_b == y_true)))  # A wrong, B right
    d10 = int(np.sum((y_a == y_true) & (y_b != y_true)))  # A right, B wrong
    n = d01 + d10
    if n == 0:
        return 1.0
    return float(binomtest(min(d01, d10), n, 0.5, alternative="two-sided").pvalue)
