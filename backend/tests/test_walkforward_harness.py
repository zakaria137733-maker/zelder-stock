"""Offline tests for the walk-forward harness (services/walkforward.py)."""

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
    assert set(result["overall"]) == {"majority_prior", "momentum", "logistic"}
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
