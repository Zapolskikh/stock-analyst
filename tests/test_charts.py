"""
Tests for src/charts/price_chart.py and src/charts/fundamental_chart.py

No network — uses synthetic DataFrames.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.charts.price_chart import build_price_chart, _calc_rsi, _calc_macd
from src.charts.fundamental_chart import build_fundamental_charts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ohlcv(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = pd.Series(range(100, 100 + n), dtype=float)
    return pd.DataFrame(
        {
            "Open":   close - 1,
            "High":   close + 2,
            "Low":    close - 2,
            "Close":  close,
            "Volume": [500_000] * n,
        },
        index=idx,
    )


def _annual_df(years: list[int], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "end":   pd.to_datetime([f"{y}-12-31" for y in years]),
            "start": pd.to_datetime([f"{y}-01-01" for y in years]),
            "val":   values,
            "form":  ["10-K"] * len(years),
            "filed": pd.to_datetime([f"{y+1}-02-15" for y in years]),
            "accn":  [f"acc{i}" for i in range(len(years))],
        }
    )


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def test_calc_rsi_length():
    close = pd.Series(range(1, 51), dtype=float)
    rsi = _calc_rsi(close, period=14)
    assert len(rsi) == len(close)


def test_calc_rsi_range():
    close = pd.Series(range(1, 51), dtype=float)
    rsi = _calc_rsi(close, period=14).dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_calc_macd_returns_three_series():
    close = pd.Series(range(1, 51), dtype=float)
    macd, signal, hist = _calc_macd(close)
    assert len(macd) == len(signal) == len(hist) == len(close)


# ---------------------------------------------------------------------------
# build_price_chart
# ---------------------------------------------------------------------------

def test_build_price_chart_returns_figure():
    fig = build_price_chart(_ohlcv(), "TEST")
    assert isinstance(fig, go.Figure)


def test_build_price_chart_has_four_rows():
    fig = build_price_chart(_ohlcv(), "TEST")
    # 4-panel layout: candlestick, volume, RSI, MACD
    rows = {t.yaxis for t in fig.data}
    assert len(rows) >= 4


def test_build_price_chart_title_contains_ticker():
    fig = build_price_chart(_ohlcv(), "NVDA")
    title_text = fig.layout.annotations[0].text if fig.layout.annotations else ""
    assert "NVDA" in title_text


# ---------------------------------------------------------------------------
# build_fundamental_charts
# ---------------------------------------------------------------------------

def _make_fundamentals() -> dict:
    years = [2020, 2021, 2022, 2023]
    return {
        "revenue":          _annual_df(years, [200e9, 250e9, 300e9, 350e9]),
        "gross_profit":     _annual_df(years, [100e9, 130e9, 160e9, 190e9]),
        "operating_income": _annual_df(years, [60e9,  80e9,  100e9, 120e9]),
        "net_income":       _annual_df(years, [50e9,  65e9,  80e9,  95e9]),
        "operating_cf":     _annual_df(years, [70e9,  90e9,  110e9, 130e9]),
        "capex":            _annual_df(years, [10e9,  12e9,  14e9,  16e9]),
        "equity":           _annual_df(years, [80e9,  90e9,  100e9, 110e9]),
        "total_liabilities":_annual_df(years, [200e9, 210e9, 220e9, 230e9]),
        "long_term_debt":   _annual_df(years, [90e9,  85e9,  80e9,  75e9]),
        "eps_diluted":      _annual_df(years, [5.0,   6.5,   8.0,   9.5]),
        "eps_basic":        _annual_df(years, [5.1,   6.6,   8.1,   9.6]),
    }


def test_build_fundamental_charts_returns_list(tmp_path: Path):
    charts = build_fundamental_charts(_make_fundamentals(), "TEST", tmp_path)
    assert isinstance(charts, list)
    assert len(charts) > 0


def test_build_fundamental_charts_files_are_created(tmp_path: Path):
    charts = build_fundamental_charts(_make_fundamentals(), "TEST", tmp_path)
    for name, path in charts:
        assert Path(path).exists(), f"Chart file missing: {path}"


def test_build_fundamental_charts_html_output(tmp_path: Path):
    charts = build_fundamental_charts(_make_fundamentals(), "TEST", tmp_path)
    for name, path in charts:
        content = Path(path).read_text(encoding="utf-8")
        assert "<html" in content.lower() or "plotly" in content.lower()


def test_build_fundamental_charts_empty_fundamentals_ok(tmp_path: Path):
    # Should not raise even with no data
    charts = build_fundamental_charts({}, "EMPTY", tmp_path)
    assert isinstance(charts, list)
