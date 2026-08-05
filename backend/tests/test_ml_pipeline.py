"""ML serving pipeline: raw feature construction, scaler application, and graceful
fallback when no model artifact is deployed.

These tests exercise the exact code path behind /api/predictions but never need a
live InfluxDB or a trained model file.
"""

import pytest

pytest.importorskip("torch")

import numpy as np

from routers import predictions as preds
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
    monkeypatch.setattr(preds, "SCALER_PATH", str(tmp_path / "scaler.json"))
    monkeypatch.setattr(preds, "MODEL_DIR", str(tmp_path))
    out = preds.ensemble_predict("AAPL", _rows(15))
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
    acc, probs = result
    assert 0.0 <= acc <= 1.0
    assert probs.shape == y.shape
    assert np.all((probs >= 0) & (probs <= 1))
