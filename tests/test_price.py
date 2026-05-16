"""
Tests for src/data/price.py

All yfinance calls are mocked — no network required.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from src.data.price import fetch_dividends, fetch_info, fetch_ohlcv, fetch_splits


def _make_ohlcv(n: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "Open":   [100.0] * n,
            "High":   [105.0] * n,
            "Low":    [98.0]  * n,
            "Close":  [102.0] * n,
            "Volume": [1_000_000] * n,
            "Dividends": [0.0] * n,
            "Stock Splits": [0.0] * n,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# fetch_ohlcv
# ---------------------------------------------------------------------------

@patch("src.data.price.yf.Ticker")
def test_fetch_ohlcv_returns_five_columns(mock_ticker_cls):
    mock_ticker_cls.return_value.history.return_value = _make_ohlcv()
    df = fetch_ohlcv("AAPL", period="5y")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 10


@patch("src.data.price.yf.Ticker")
def test_fetch_ohlcv_index_is_timezone_naive(mock_ticker_cls):
    mock_ticker_cls.return_value.history.return_value = _make_ohlcv()
    df = fetch_ohlcv("AAPL")
    assert df.index.tz is None


@patch("src.data.price.yf.Ticker")
def test_fetch_ohlcv_raises_on_empty_response(mock_ticker_cls):
    mock_ticker_cls.return_value.history.return_value = pd.DataFrame()
    with pytest.raises(ValueError, match="No price data"):
        fetch_ohlcv("FAKE")


# ---------------------------------------------------------------------------
# fetch_dividends
# ---------------------------------------------------------------------------

@patch("src.data.price.yf.Ticker")
def test_fetch_dividends_returns_series(mock_ticker_cls):
    idx = pd.date_range("2023-01-01", periods=4, freq="QE", tz="UTC")
    mock_ticker_cls.return_value.dividends = pd.Series([0.23, 0.23, 0.24, 0.24], index=idx)
    result = fetch_dividends("AAPL")
    assert isinstance(result, pd.Series)
    assert len(result) == 4
    assert result.index.tz is None


@patch("src.data.price.yf.Ticker")
def test_fetch_dividends_empty_is_ok(mock_ticker_cls):
    mock_ticker_cls.return_value.dividends = pd.Series([], dtype=float)
    result = fetch_dividends("NODIVY")
    assert result.empty


# ---------------------------------------------------------------------------
# fetch_splits
# ---------------------------------------------------------------------------

@patch("src.data.price.yf.Ticker")
def test_fetch_splits_timezone_stripped(mock_ticker_cls):
    idx = pd.DatetimeIndex(["2020-08-31"], tz="UTC")
    mock_ticker_cls.return_value.splits = pd.Series([4.0], index=idx)
    result = fetch_splits("AAPL")
    assert result.index.tz is None
    assert result.iloc[0] == 4.0


# ---------------------------------------------------------------------------
# fetch_info
# ---------------------------------------------------------------------------

@patch("src.data.price.yf.Ticker")
def test_fetch_info_returns_dict(mock_ticker_cls):
    mock_ticker_cls.return_value.info = {"sector": "Technology", "marketCap": 3e12}
    info = fetch_info("AAPL")
    assert isinstance(info, dict)
    assert info["sector"] == "Technology"


@patch("src.data.price.yf.Ticker")
def test_fetch_info_returns_empty_dict_on_none(mock_ticker_cls):
    mock_ticker_cls.return_value.info = None
    info = fetch_info("FAKE")
    assert info == {}
