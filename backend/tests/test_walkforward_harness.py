"""Offline tests for the walk-forward harness (services/walkforward.py)."""

import numpy as np
import pytest

from services import walkforward as wf


def test_trailing_return_causal():
    prices = [100.0, 105.0, 110.0]
    assert wf.trailing_return(prices, 2, 1) == pytest.approx((110.0 - 105.0) / 105.0 * 100)
    assert wf.trailing_return(prices, 0, 1) == 0.0  # no prior bar


def test_binary_labels_up_down_flat():
    up = [100.0 + i for i in range(30)]
    down = [300.0 - i for i in range(30)]
    flat = [100.0] * 30
    assert all(y == 1 for _, y in wf.binary_labels(up))
    assert all(y == 0 for _, y in wf.binary_labels(down))
    assert wf.binary_labels(flat) == []


def test_majority_prior_perfect_on_all_up():
    prices = [100.0 + i for i in range(60)]
    result = wf.evaluate(prices, n_folds=5, min_train_windows=4)
    assert result is not None
    assert result["overall"]["majority_prior"] == pytest.approx(1.0)


def test_momentum_beats_coin_flip_on_autocorrelated_series():
    prices = wf.synthetic_prices(n=1000, seed=7, momentum=0.3)
    result = wf.evaluate(prices, n_folds=5)
    assert result is not None
    assert result["overall"]["momentum"] > 0.52
    assert result["coin_flip_expected"] == 0.5


def test_walk_forward_shape_and_sane_values():
    prices = wf.synthetic_prices(n=800, seed=123)
    result = wf.evaluate(prices, n_folds=5)
    assert result is not None
    assert result["n_folds_run"] == 4
    assert len(result["folds"]) == 4
    expected = {"majority_prior", "momentum", "logistic", "mlp"}
    if wf._xgboost_available():
        expected.add("xgboost")
    assert set(result["overall"]) == expected
    for value in result["overall"].values():
        assert 0.0 <= value <= 1.0


def test_logistic_not_worse_than_momentum_on_autocorrelated_series():
    # with real return autocorrelation, a model given trailing returns should
    # not be systematically worse than the momentum continuation baseline
    prices = wf.synthetic_prices(n=1200, seed=9, momentum=0.25)
    result = wf.evaluate(prices, n_folds=5)
    assert result is not None
    assert result["overall"]["logistic"] + 0.03 >= result["overall"]["momentum"]


def test_evaluate_none_when_insufficient_data():
    assert wf.evaluate([100.0] * 15) is None


def test_models_filter_restricts_methods():
    prices = wf.synthetic_prices(n=800, seed=11)
    result = wf.evaluate(prices, n_folds=5, models=["majority_prior", "momentum"])
    assert result is not None
    assert set(result["overall"]) == {"majority_prior", "momentum"}


def test_mlp_sane_and_not_worse_than_coin_flip():
    prices = wf.synthetic_prices(n=1200, seed=13, momentum=0.25)
    result = wf.evaluate(prices, n_folds=5, models=["mlp"])
    assert result is not None
    assert result["overall"]["mlp"] > 0.5


def test_xgboost_present_and_sane_when_installed():
    pytest.importorskip("xgboost")
    prices = wf.synthetic_prices(n=1200, seed=15, momentum=0.25)
    result = wf.evaluate(prices, n_folds=5, models=["xgboost"])
    assert result is not None
    assert result["overall"]["xgboost"] > 0.5


def test_threshold_changes_window_count():
    prices = wf.synthetic_prices(n=800, seed=17)
    loose = wf.evaluate(prices, n_folds=5, threshold_pct=0.5)
    tight = wf.evaluate(prices, n_folds=5, threshold_pct=3.0)
    assert loose is not None and tight is not None
    assert loose["n_windows_total"] > tight["n_windows_total"]


def test_evaluate_oof_returns_time_ordered_predictions():
    prices = wf.synthetic_prices(n=800, seed=11)
    oof = wf.evaluate_oof(prices, n_folds=5, models=["momentum", "majority_prior"])
    assert oof is not None
    for name in ("momentum", "majority_prior"):
        assert len(oof[name]["pred"]) == len(oof[name]["true"]) > 0
        assert all(y in (0, 1) for y in oof[name]["true"])
    # pooled OOF accuracy equals re-computing accuracy over the pooled predictions
    pooled = wf.evaluate_oof(prices, n_folds=5, models=["momentum"])
    acc = float((np.asarray(pooled["momentum"]["pred"]) == np.asarray(pooled["momentum"]["true"])).mean())
    assert 0.0 < acc < 1.0
    # and matches the mean fold accuracy from evaluate() for a balanced split
    summary = wf.evaluate(prices, n_folds=5, models=["momentum"])
    assert acc == pytest.approx(summary["overall"]["momentum"], abs=0.05)


def test_evaluate_oof_none_when_insufficient():
    assert wf.evaluate_oof([100.0] * 15) is None


def test_evaluate_oof_fold_indexed_output():
    prices = wf.synthetic_prices(n=800, seed=19)
    oof = wf.evaluate_oof(prices, n_folds=5, models=["momentum"])
    assert oof is not None
    folds = oof["momentum"]["folds"]
    assert folds, "expected per-fold entries"
    fold_nums = [f["fold"] for f in folds]
    assert fold_nums == sorted(fold_nums)
    per_fold = [x for f in folds for x in f["pred"]]
    assert len(per_fold) == len(oof["momentum"]["pred"]) == len(oof["momentum"]["true"]) > 0
    assert np.allclose(per_fold, oof["momentum"]["pred"])
    # pooled fold arrays equal the pooled top-level arrays
    pooled = wf.oof_pooled(oof, "momentum")
    assert np.allclose(pooled["pred"], oof["momentum"]["pred"])
    assert np.allclose(pooled["true"], oof["momentum"]["true"])


def test_split_holdout_and_gated_metrics():
    prices = wf.synthetic_prices(n=800, seed=19)
    oof = wf.evaluate_oof(prices, n_folds=5, models=["momentum", "majority_prior"])
    assert oof is not None
    fit, hold = wf.split_holdout(oof, "momentum")
    assert fit and hold
    assert hold is oof["momentum"]["folds"][-1]
    fit_ids = [f["fold"] for f in fit]
    fit_pool = wf.oof_pooled(oof, "momentum", folds=fit_ids)
    hold_pool = wf.oof_pooled(oof, "momentum", folds=[hold["fold"]])
    assert len(fit_pool["true"]) + len(hold_pool["true"]) == len(oof["momentum"]["true"])
    # gate closed → every holdout window is NO_SIGNAL, gated accuracy is None
    acc, share = wf.holdout_gated_metrics(hold_pool["prob"], hold_pool["true"], 0.5, 0.5, gate=False)
    assert acc is None
    assert share == 1.0
    # gate open with a perfect threshold → all windows directional and accurate
    y = hold_pool["true"]
    probs = np.asarray(y, dtype=float)
    acc, share = wf.holdout_gated_metrics(probs, y, buy_threshold=0.0, sell_threshold=0.0, gate=True)
    assert acc == 1.0
    assert share == 0.0
    # split_holdout returns (None, None) for a method without folds
    assert wf.split_holdout({"m": {"folds": []}}, "m") == (None, None)


def test_mcnemar_pvalue_identical_is_not_significant():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    assert wf.mcnemar_pvalue(y, y, y) == 1.0


def test_mcnemar_pvalue_detects_systematic_difference():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    perfect = y
    anti = 1 - y
    assert wf.mcnemar_pvalue(y, perfect, anti) < 0.05


def test_binomial_ci_sane_bounds():
    lo, hi = wf.binomial_ci(100, 50)
    assert lo <= 0.5 <= hi
    assert 0.0 <= lo <= hi <= 1.0
    assert wf.binomial_ci(0, 0) == (0.0, 0.0)
