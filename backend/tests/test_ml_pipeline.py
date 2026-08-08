"""ML serving pipeline: raw feature construction, scaler application, and graceful
fallback when no model artifact is deployed.

These tests exercise the exact code path behind /api/predictions but never need a
live InfluxDB or a trained model file.
"""

import pytest

pytest.importorskip("torch")

import numpy as np

from services import lstm_predictor as lsp


def _rows(n, start=100.0):
    rows = []
    for i in range(n):
        rows.append({
            "time": f"2026-01-{i + 1:02d}",
            "price": start + i,
            "volume": 1000.0,
            "sentiment": 60.0,
            "spy_ret": 0.1,
            "vix": 18.0,
        })
    return rows


def test_build_raw_features_shape():
    w = lsp.build_raw_features(_rows(15))
    assert w is not None
    assert w.shape == (lsp.SEQUENCE_LEN, lsp.FEATURES)
    # feature 0 is sentiment / 100
    assert np.isclose(w[0, 0], 0.6)


def test_build_raw_features_uses_last_10_rows():
    rows = _rows(15)
    w = lsp.build_raw_features(rows)
    # first window row has no prior bar, so its price change is 0
    assert np.isclose(w[0, 1], 0.0)
    # later rows have a positive price change (price increases by 1 per row)
    assert w[-1, 1] > 0
    assert np.isclose(w[-1, 0], 0.6)


def test_build_raw_features_insufficient_data():
    assert lsp.build_raw_features(_rows(3)) is None


def test_apply_scaler_identity_when_absent():
    w = np.ones((2, 10, 12), dtype=np.float32)
    out = lsp.apply_scaler(w, {})
    assert out.shape == w.shape
    assert np.allclose(out, w)


def test_apply_scaler_scales_per_feature():
    scaler = {"0": {"min": 0.0, "range": 100.0}, "5": {"min": 20.0, "range": 40.0}}
    w = np.zeros((1, 1, 12), dtype=np.float32)
    w[0, 0, 0] = 50.0
    w[0, 0, 5] = 40.0
    out = lsp.apply_scaler(w, scaler)
    assert np.isclose(out[0, 0, 0], 0.5)
    assert np.isclose(out[0, 0, 5], 0.5)


def test_apply_scaler_clips_extremes():
    scaler = {"0": {"min": 0.0, "range": 1.0}}
    w = np.zeros((1, 1, 12), dtype=np.float32)
    w[0, 0, 0] = 1000.0
    out = lsp.apply_scaler(w, scaler)
    assert out[0, 0, 0] == 5.0


def test_apply_scaler_unscaled_features_unchanged():
    # features not present in the scaler keep their raw value (identity)
    w = np.full((1, 1, 12), 3.0, dtype=np.float32)
    scaler = {"3": {"min": 0.0, "range": 10.0}}
    out = lsp.apply_scaler(w, scaler)
    assert np.isclose(out[0, 0, 1], 3.0)
    assert np.isclose(out[0, 0, 3], 0.3)


def test_daily_returns():
    m = {"2026-01-01": 100.0, "2026-01-02": 110.0, "2026-01-03": 99.0}
    r = lsp._daily_returns(m)
    assert r["2026-01-01"] == 0.0
    assert np.isclose(r["2026-01-02"], 10.0)
    assert np.isclose(r["2026-01-03"], -10.0)


def test_build_eval_sequences_matches_label_convention():
    # monotonic +1/day prices → forward return over horizon 5 is well above +1%
    rows = _rows(30)
    X, y = lsp.build_eval_sequences(rows, horizon=5)
    assert len(X) == len(y)
    assert X.shape[1:] == (lsp.SEQUENCE_LEN, lsp.FEATURES)
    assert set(np.unique(y)).issubset({0, 1})
    assert (y == 1).all()  # all windows moved up by more than 1%


def test_build_eval_sequences_insufficient_data():
    X, y = lsp.build_eval_sequences(_rows(10), horizon=5)
    assert len(X) == 0
    assert len(y) == 0


def test_ensemble_predict_graceful_when_not_trained(monkeypatch, tmp_path):
    monkeypatch.setattr(lsp, "SCALER_PATH", str(tmp_path / "scaler.json"))
    monkeypatch.setattr(lsp, "MODEL_DIR", str(tmp_path))
    out = lsp.predict_ensemble("AAPL", _rows(15))
    assert out["signal"] == "HOLD"
    assert "Model not trained" in out["error"]
    assert out["prob_up"] == 0.5


def test_evaluate_deployed_returns_none_without_models(monkeypatch, tmp_path):
    monkeypatch.setattr(lsp, "SCALER_PATH", str(tmp_path / "scaler.json"))
    monkeypatch.setattr(lsp, "MODEL_DIR", str(tmp_path))
    X = np.zeros((3, 10, 12), dtype=np.float32)
    y = np.array([1, 0, 1])
    assert lsp.evaluate_deployed(X, y) is None


def test_deployed_artifact_scores_on_synthetic_data(monkeypatch):
    """Score the model artifact that ships with the repo (lstm_model.pt + scaler.json)."""
    import os

    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    assert os.path.exists(os.path.join(models_dir, "lstm_model.pt")), "expected committed model artifact"
    monkeypatch.setattr(lsp, "MODEL_DIR", models_dir)
    monkeypatch.setattr(lsp, "MODEL_PATH", os.path.join(models_dir, "lstm_model.pt"))
    monkeypatch.setattr(lsp, "SCALER_PATH", os.path.join(models_dir, "scaler.json"))

    rows = _rows(30)
    X, y = lsp.build_eval_sequences(rows, horizon=5)
    result = lsp.evaluate_deployed(X, y)
    assert result is not None, "deployed artifact must produce predictions"
    metrics, probs = result
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert metrics["auc"] is None or 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["majority_baseline"] <= 1.0
    assert probs.shape == y.shape
    assert np.all((probs >= 0) & (probs <= 1))


def test_build_sequences_matches_build_raw_features_convention():
    # A window ends at bar i and its label is the forward return from bar i,
    # exactly like build_eval_sequences/build_raw_features (no 1-bar offset).
    rows = [{"ticker": "AAPL", "hour": f"2026-01-{i + 1:02d}", "price": 100.0 + i,
             "volume": 1000.0, "sentiment": 60.0, "spy_ret": 0.1, "vix": 18.0}
            for i in range(40)]
    X, y = lsp.build_sequences(rows, horizon=5)
    assert X.shape[1:] == (lsp.SEQUENCE_LEN, lsp.FEATURES)
    assert len(X) == len(y)
    assert (y == 1).all()  # monotonically rising prices → all windows up


def test_build_sequences_temporal_split_is_leak_free():
    # Windows sorted by end time; train must be strictly older than val.
    rows = []
    for t in range(60):
        rows.append({"ticker": "AAPL", "hour": f"2026-01-{t + 1:02d}",
                     "price": 100.0 + (0.5 if t % 2 == 0 else -0.5) * (t % 3),
                     "volume": 1000.0, "sentiment": 60.0, "spy_ret": 0.0, "vix": 18.0})
    X, y, times = lsp.build_sequences(rows, horizon=5, return_times=True)
    order = np.argsort(np.asarray(times), kind="stable")
    split = int(0.8 * len(order))
    assert order[:split].max() < order[split:].min()  # preserved order → temporal
    assert len(X) == len(times)
    assert X.shape[1:] == (lsp.SEQUENCE_LEN, lsp.FEATURES)


def test_classification_metrics_reports_auc_and_baseline():
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    probs = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.6, 0.4])
    m = lsp.classification_metrics(y, probs)
    assert m["n"] == 8
    assert m["auc"] is not None and 0.0 <= m["auc"] <= 1.0
    assert 0.0 <= m["balanced_accuracy"] <= 1.0
    assert m["majority_baseline"] == pytest.approx(0.5)


def _varying_rows(n, price=100.0):
    rows = []
    for i in range(n):
        price = price * (1.0 + 0.01 * np.sin(i / 3.0) + 0.0015 * (i % 5))
        rows.append({
            "ticker": "AAPL",
            "hour": f"2026-01-{i:04d}",
            "time": f"2026-01-{i:04d}",
            "price": price,
            "volume": 1_000_000.0 + (i % 7) * 100_000.0,
            "sentiment": 40.0 + (i % 9),
            "spy_ret": 0.1 * (i % 3),
            "vix": 16.0 + (i % 5),
        })
    return rows


def test_train_and_serve_windows_are_identical():
    """build_sequences() and build_raw_features() must produce the exact same
    window for the same end bar, even when the series is longer than LOOKBACK
    (the case that used to let cumulative indicators like OBV drift between
    training and serving)."""
    rows = _varying_rows(lsp.LOOKBACK + 60)
    X, y, times = lsp.build_sequences(rows, horizon=lsp.HORIZON, return_times=True)
    assert len(X) > 0
    by_time = {}
    for t, x in zip(times, X, strict=True):
        by_time.setdefault(t, []).append(x)

    for i in range(lsp.SEQUENCE_LEN, len(rows) - lsp.HORIZON):
        t = rows[i]["time"]
        if t not in by_time:
            continue  # neutral window (no label)
        raw = lsp.build_raw_features(rows[: i + 1])
        assert raw is not None
        for w in by_time[t]:
            assert np.array_equal(raw, w), f"feature drift at end bar {i}"


def test_build_raw_features_longer_than_lookback_uses_trailing_context():
    rows = _varying_rows(lsp.LOOKBACK + 20)
    # last window via full-length series must equal last window via explicit tail
    full = lsp.build_raw_features(rows)
    tail = lsp.build_raw_features(rows[-lsp.LOOKBACK:])
    assert full is not None and tail is not None
    assert np.allclose(full, tail)


def test_fit_scaler_min_max_per_feature():
    X = np.array([[[0.0, 10.0], [5.0, 30.0], [10.0, 20.0]]], dtype=np.float32)
    s = lsp.fit_scaler(X)
    assert s[0] == {"min": 0.0, "range": 10.0}
    assert s[1] == {"min": 10.0, "range": 20.0}


def test_fit_scaler_constant_feature_uses_unit_range():
    X = np.zeros((2, 2, 1), dtype=np.float32)
    s = lsp.fit_scaler(X)
    assert s[0] == {"min": 0.0, "range": 1.0}


def test_youden_threshold_on_perfectly_separable_probs():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    t = lsp.youden_threshold(y, p)
    assert 0.4 <= t <= 0.6


def test_youden_threshold_degenerate():
    assert lsp.youden_threshold(np.array([1, 1, 1]), np.array([0.5, 0.5, 0.5])) == 0.5


def test_signal_from_prob_requires_gate():
    th = {"AAPL": {"gate": False, "buy_threshold": 0.6, "sell_threshold": 0.4}}
    assert lsp.signal_from_prob(0.9, "AAPL", th) == "NO_SIGNAL"
    th["AAPL"]["gate"] = True
    assert lsp.signal_from_prob(0.61, "AAPL", th) == "BUY"
    assert lsp.signal_from_prob(0.39, "AAPL", th) == "SELL"
    assert lsp.signal_from_prob(0.5, "AAPL", th) == "NO_SIGNAL"


def test_signal_from_prob_without_thresholds_is_no_signal():
    assert lsp.signal_from_prob(0.99, "AAPL", {}) == "NO_SIGNAL"


def test_prediction_evidence_subset():
    meta = {
        "n_windows": 100, "lstm_acc": 0.55, "momentum_acc": 0.5, "auc": 0.58,
        "p_vs_momentum": 0.03, "buy_threshold": 0.6, "sell_threshold": 0.4,
        "gate": True, "internal_only": "dropped",
    }
    ev = lsp.prediction_evidence(meta)
    assert "gate" not in ev
    assert "internal_only" not in ev
    assert ev["lstm_acc"] == 0.55
    assert ev["p_vs_momentum"] == 0.03
