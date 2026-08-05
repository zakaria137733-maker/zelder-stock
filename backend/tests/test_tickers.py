"""Ticker validation shared by every router that touches the price/sentiment data."""

import pytest
from fastapi import HTTPException

from tickers import TICKERS, validate_ticker


def test_known_tickers_validate():
    for t in TICKERS:
        assert validate_ticker(t) == t


def test_lowercase_is_normalized():
    assert validate_ticker("aapl") == "AAPL"


def test_unknown_ticker_raises_404():
    with pytest.raises(HTTPException) as exc:
        validate_ticker("BTC")
    assert exc.value.status_code == 404


def test_unknown_ticker_message():
    try:
        validate_ticker("DOGE")
    except HTTPException as e:
        assert "DOGE" in e.detail
