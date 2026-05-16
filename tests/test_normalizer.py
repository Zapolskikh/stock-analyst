"""
Tests for src/data/normalizer.py — fully offline, no network.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.data.normalizer import (
    NormalisedData,
    _align,
    _compute_ttm_flow,
    _data_quality,
    _pct_change,
    _pct_ratio,
    _quarterly_series,
    _ratio,
    _subtract,
    _valid,
    normalise,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_annual_df(years: list[int], values: list[float]) -> pd.DataFrame:
    """Build a minimal EDGAR-style annual DataFrame."""
    return pd.DataFrame(
        {
            "end":   pd.to_datetime([f"{y}-12-31" for y in years]),
            "start": pd.to_datetime([f"{y}-01-01" for y in years]),
            "val":   values,
            "form":  ["10-K"] * len(years),
            "filed": pd.to_datetime([f"{y + 1}-02-01" for y in years]),
            "accn":  [f"acc{i}" for i in range(len(years))],
        }
    )


def _make_ohlcv(n: int = 10, last_close: float = 150.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = [last_close] * n
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": [1_000_000] * n},
        index=idx,
    )


def _base_fundamentals() -> dict[str, pd.DataFrame]:
    years = [2020, 2021, 2022, 2023]
    return {
        "revenue":           _make_annual_df(years, [200e9, 250e9, 300e9, 350e9]),
        "gross_profit":      _make_annual_df(years, [100e9, 130e9, 160e9, 200e9]),
        "operating_income":  _make_annual_df(years, [50e9,  70e9,  90e9,  110e9]),
        "net_income":        _make_annual_df(years, [40e9,  55e9,  70e9,  85e9]),
        "operating_cf":      _make_annual_df(years, [60e9,  80e9,  100e9, 120e9]),
        "capex":             _make_annual_df(years, [10e9,  12e9,  14e9,  16e9]),
        "equity":            _make_annual_df(years, [80e9,  90e9,  100e9, 110e9]),
        "total_assets":      _make_annual_df(years, [200e9, 220e9, 240e9, 260e9]),
        "total_liabilities": _make_annual_df(years, [120e9, 130e9, 140e9, 150e9]),
        "long_term_debt":    _make_annual_df(years, [50e9,  45e9,  40e9,  35e9]),
        "eps_diluted":       _make_annual_df(years, [5.0,   7.0,   9.0,   11.0]),
        "rd_expense":        _make_annual_df(years, [5e9,   6e9,   7e9,   8e9]),
    }


def _base_info() -> dict:
    return {
        "currentPrice":  155.0,
        "marketCap":     2.4e12,
        "trailingPE":    28.5,
        "forwardPE":     24.0,
        "beta":          1.2,
        "sector":        "Technology",
        "industry":      "Semiconductors",
        "dividendYield": 0.005,
    }


# ---------------------------------------------------------------------------
# Unit tests — helpers
# ---------------------------------------------------------------------------

class TestValidHelper:
    def test_finite_number(self):
        assert _valid(3.14)

    def test_nan_is_not_valid(self):
        assert not _valid(float("nan"))

    def test_inf_is_not_valid(self):
        assert not _valid(float("inf"))

    def test_none_is_not_valid(self):
        assert not _valid(None)  # type: ignore[arg-type]


class TestAlign:
    def test_full_overlap(self):
        result = _align([2020, 2021, 2022], [2020, 2021, 2022], [1.0, 2.0, 3.0])
        assert result == [1.0, 2.0, 3.0]

    def test_missing_year_becomes_nan(self):
        result = _align([2020, 2021, 2022], [2020, 2022], [1.0, 3.0])
        assert result[0] == 1.0
        assert math.isnan(result[1])
        assert result[2] == 3.0

    def test_empty_source_all_nan(self):
        result = _align([2020, 2021], [], [])
        assert all(math.isnan(v) for v in result)


class TestPctChange:
    def test_first_element_is_nan(self):
        result = _pct_change([100.0, 120.0, 150.0])
        assert math.isnan(result[0])

    def test_correct_growth(self):
        result = _pct_change([100.0, 125.0])
        assert abs(result[1] - 25.0) < 1e-9

    def test_negative_base(self):
        result = _pct_change([-100.0, -50.0])
        assert abs(result[1] - 50.0) < 1e-9  # improvement of 50 %

    def test_zero_base_gives_nan(self):
        result = _pct_change([0.0, 100.0])
        assert math.isnan(result[1])

    def test_single_element(self):
        result = _pct_change([42.0])
        assert len(result) == 1
        assert math.isnan(result[0])


class TestRatioHelpers:
    def test_pct_ratio_basic(self):
        result = _pct_ratio([50.0, 75.0], [100.0, 150.0])
        assert result == [50.0, 50.0]

    def test_ratio_zero_denominator(self):
        result = _ratio([10.0], [0.0])
        assert math.isnan(result[0])

    def test_subtract_basic(self):
        assert _subtract([100.0, 200.0], [30.0, 50.0]) == [70.0, 150.0]

    def test_subtract_nan_propagates(self):
        result = _subtract([100.0, float("nan")], [30.0, 50.0])
        assert result[0] == 70.0
        assert math.isnan(result[1])


class TestDataQuality:
    def test_good_when_4_years_and_no_missing_core(self):
        assert _data_quality(4, []) == "good"

    def test_good_with_non_core_missing(self):
        assert _data_quality(4, ["rd_expense", "inventory"]) == "good"

    def test_partial_when_2_years_one_core_missing(self):
        assert _data_quality(2, ["operating_cf"]) == "partial"

    def test_poor_when_1_year(self):
        assert _data_quality(1, []) == "poor"

    def test_poor_when_many_core_missing(self):
        assert _data_quality(5, ["revenue", "net_income", "operating_cf"]) == "poor"


# ---------------------------------------------------------------------------
# Integration tests — normalise()
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_returns_normalised_data_instance(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info(), "TEST")
        assert isinstance(nd, NormalisedData)

    def test_ticker_stored(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info(), "aapl")
        assert nd.ticker == "AAPL"

    def test_years_populated(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        assert nd.years == [2020, 2021, 2022, 2023]

    def test_years_of_history(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        assert nd.years_of_history == 4

    def test_data_quality_good(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        assert nd.data_quality == "good"

    # --- Raw series ---------------------------------------------------------

    def test_revenue_annual(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        assert nd.revenue_annual == [200e9, 250e9, 300e9, 350e9]

    def test_net_income_annual(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        assert nd.net_income_annual == [40e9, 55e9, 70e9, 85e9]

    # --- Derived: FCF -------------------------------------------------------

    def test_fcf_annual_correct(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        expected = [50e9, 68e9, 86e9, 104e9]
        for got, exp in zip(nd.fcf_annual, expected):
            assert abs(got - exp) < 1.0

    # --- Derived: margins ---------------------------------------------------

    def test_gross_margin_correct(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        # gross_profit / revenue * 100 for 2020: 100/200 = 50 %
        assert abs(nd.gross_margin_annual[0] - 50.0) < 1e-6

    def test_net_margin_correct(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        # 40/200 = 20 %
        assert abs(nd.net_margin_annual[0] - 20.0) < 1e-6

    def test_operating_margin_correct(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        # 50/200 = 25 %
        assert abs(nd.operating_margin_annual[0] - 25.0) < 1e-6

    # --- Derived: growth rates ---------------------------------------------

    def test_revenue_growth_first_element_nan(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        assert math.isnan(nd.revenue_growth_annual[0])

    def test_revenue_growth_2021(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        # 250→200 = +25 %
        assert abs(nd.revenue_growth_annual[1] - 25.0) < 1e-6

    def test_eps_growth_populated(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        assert len(nd.eps_growth_annual) == 4

    # --- Derived: ROE / ROA / D/E ------------------------------------------

    def test_roe_correct(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        # 40/80 * 100 = 50 %
        assert abs(nd.roe_annual[0] - 50.0) < 1e-6

    def test_roa_correct(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        # 40/200 * 100 = 20 %
        assert abs(nd.roa_annual[0] - 20.0) < 1e-6

    def test_debt_to_equity_correct(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        # 50/80 ≈ 0.625
        assert abs(nd.debt_to_equity_annual[0] - 50e9 / 80e9) < 1e-9

    # --- Scalars from info -------------------------------------------------

    def test_current_price_from_info(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        assert nd.current_price == 155.0

    def test_current_price_fallback_from_ohlcv(self):
        info = {k: v for k, v in _base_info().items() if k != "currentPrice"}
        info.pop("regularMarketPrice", None)
        nd = normalise(_base_fundamentals(), _make_ohlcv(last_close=123.0), info)
        assert nd.current_price == 123.0

    def test_sector_stored(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        assert nd.sector == "Technology"

    def test_pe_trailing(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        assert nd.pe_trailing == 28.5

    def test_beta(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        assert nd.beta == 1.2

    # --- Partial data -------------------------------------------------------

    def test_missing_capex_fcf_is_nan(self):
        funds = _base_fundamentals()
        del funds["capex"]
        nd = normalise(funds, _make_ohlcv(), _base_info())
        assert all(math.isnan(v) for v in nd.fcf_annual)

    def test_missing_equity_roe_is_nan(self):
        funds = _base_fundamentals()
        del funds["equity"]
        nd = normalise(funds, _make_ohlcv(), _base_info())
        assert all(math.isnan(v) for v in nd.roe_annual)

    def test_partial_data_quality(self):
        funds = {k: v for k, v in _base_fundamentals().items()
                 if k in ("revenue", "net_income")}
        nd = normalise(funds, _make_ohlcv(), _base_info())
        assert nd.data_quality in ("partial", "poor")

    def test_empty_fundamentals_returns_poor_quality(self):
        nd = normalise({}, _make_ohlcv(), _base_info())
        assert nd.data_quality == "poor"
        assert nd.years == []

    # --- Series lengths all equal ------------------------------------------

    def test_all_annual_series_same_length(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        n = nd.years_of_history
        for attr in [
            "revenue_annual", "gross_profit_annual", "net_income_annual",
            "fcf_annual", "gross_margin_annual", "net_margin_annual",
            "revenue_growth_annual", "roe_annual", "roa_annual",
            "debt_to_equity_annual", "eps_diluted_annual", "eps_growth_annual",
        ]:
            series = getattr(nd, attr)
            assert len(series) == n, f"{attr}: expected {n}, got {len(series)}"


# ---------------------------------------------------------------------------
# TTM helpers
# ---------------------------------------------------------------------------

def _make_quarterly_df(end_dates: list[str], values: list[float]) -> pd.DataFrame:
    """Build a minimal EDGAR-style quarterly DataFrame (form=10-Q).

    Sets start = end - 91 days so each row passes the single-quarter period
    filter (60–105 days) added to _quarterly_series.
    """
    ends = pd.to_datetime(end_dates)
    starts = ends - pd.Timedelta(days=91)
    return pd.DataFrame(
        {
            "end":   ends,
            "start": starts,
            "val":   values,
            "form":  ["10-Q"] * len(end_dates),
            "filed": ends,
            "accn":  [f"acc{i}" for i in range(len(end_dates))],
        }
    )


def _mixed_df(
    annual_years: list[int],
    annual_vals: list[float],
    quarterly_ends: list[str],
    quarterly_vals: list[float],
) -> pd.DataFrame:
    """Combined annual + quarterly DataFrame."""
    ann = _make_annual_df(annual_years, annual_vals)
    q   = _make_quarterly_df(quarterly_ends, quarterly_vals)
    return pd.concat([ann, q], ignore_index=True)


class TestQuarterlySeries:

    def test_returns_only_10q_rows(self):
        df = _mixed_df(
            [2022, 2023], [100.0, 200.0],
            ["2024-03-31", "2024-06-30"], [55.0, 60.0],
        )
        dates, vals = _quarterly_series(df)
        assert len(dates) == 2
        assert len(vals) == 2

    def test_sorted_oldest_first(self):
        df = _make_quarterly_df(
            ["2024-06-30", "2024-03-31", "2023-12-31", "2023-09-30"],
            [60.0, 55.0, 50.0, 45.0],
        )
        dates, vals = _quarterly_series(df)
        assert dates[0] < dates[-1]
        assert vals[0] == 45.0   # oldest = 2023-09-30

    def test_empty_dataframe_returns_empty(self):
        dates, vals = _quarterly_series(None)
        assert dates == []
        assert vals == []

    def test_annual_only_df_returns_empty(self):
        df = _make_annual_df([2021, 2022, 2023], [100.0, 200.0, 300.0])
        dates, vals = _quarterly_series(df)
        assert dates == []
        assert vals == []

    def test_dates_are_iso_strings(self):
        df = _make_quarterly_df(["2024-03-31"], [100.0])
        dates, _ = _quarterly_series(df)
        assert dates[0] == "2024-03-31"


class TestComputeTTMFlow:

    def test_sums_four_quarters(self):
        result = _compute_ttm_flow([25.0, 30.0, 35.0, 40.0])
        assert result == pytest.approx(130.0)

    def test_uses_last_four_when_more_available(self):
        # 5 values — only last 4 should be summed
        result = _compute_ttm_flow([100.0, 25.0, 30.0, 35.0, 40.0])
        assert result == pytest.approx(130.0)

    def test_returns_none_with_fewer_than_four(self):
        assert _compute_ttm_flow([25.0, 30.0, 35.0]) is None
        assert _compute_ttm_flow([]) is None

    def test_returns_none_when_nan_reduces_count(self):
        result = _compute_ttm_flow([25.0, float("nan"), 35.0, 40.0])
        # Only 3 valid values → None
        assert result is None

    def test_negative_values_included(self):
        result = _compute_ttm_flow([-10.0, -5.0, 5.0, 10.0])
        assert result == pytest.approx(0.0)


class TestNormaliseTTM:

    def _base_with_quarters(self):
        """Return fundamentals with 4 annual + 4 quarterly revenue rows."""
        funds = _base_fundamentals()
        # Add 4 quarterly rows for revenue (2024 quarters)
        q_dates = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]
        q_vals  = [90e9, 95e9, 100e9, 105e9]
        annual_df = funds["revenue"]
        quarterly_df = _make_quarterly_df(q_dates, q_vals)
        funds["revenue"] = pd.concat([annual_df, quarterly_df], ignore_index=True)
        return funds

    def test_ttm_revenue_computed_when_four_quarters(self):
        nd = normalise(self._base_with_quarters(), _make_ohlcv(), _base_info())
        assert nd.ttm_revenue is not None
        assert nd.ttm_revenue == pytest.approx(90e9 + 95e9 + 100e9 + 105e9)

    def test_ttm_revenue_none_when_no_quarterly_data(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        assert nd.ttm_revenue is None

    def test_ttm_as_of_set_when_quarterly_available(self):
        nd = normalise(self._base_with_quarters(), _make_ohlcv(), _base_info())
        assert nd.ttm_as_of == "2024-12-31"

    def test_ttm_as_of_none_when_no_quarterly(self):
        nd = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        assert nd.ttm_as_of is None

    def test_ttm_margins_derived_from_ttm_revenue(self):
        funds = _base_fundamentals()
        q_dates = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]
        # Revenue: 100 each quarter → TTM = 400
        funds["revenue"] = pd.concat([
            funds["revenue"],
            _make_quarterly_df(q_dates, [100.0, 100.0, 100.0, 100.0]),
        ], ignore_index=True)
        # Net income: 20 each quarter → TTM = 80 → margin = 20%
        funds["net_income"] = pd.concat([
            funds["net_income"],
            _make_quarterly_df(q_dates, [20.0, 20.0, 20.0, 20.0]),
        ], ignore_index=True)
        nd = normalise(funds, _make_ohlcv(), _base_info())
        assert nd.ttm_net_margin == pytest.approx(20.0)

    def test_ttm_fcf_derived(self):
        funds = _base_fundamentals()
        q_dates = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]
        funds["operating_cf"] = pd.concat([
            funds["operating_cf"],
            _make_quarterly_df(q_dates, [30e9, 30e9, 30e9, 30e9]),
        ], ignore_index=True)
        funds["capex"] = pd.concat([
            funds["capex"],
            _make_quarterly_df(q_dates, [5e9, 5e9, 5e9, 5e9]),
        ], ignore_index=True)
        nd = normalise(funds, _make_ohlcv(), _base_info())
        assert nd.ttm_fcf == pytest.approx(100e9)  # (30-5)*4

    def test_ttm_fcf_none_when_capex_insufficient(self):
        funds = _base_fundamentals()
        q_dates = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]
        # Only OCF has 4 quarters, capex has none → ttm_capex=None → ttm_fcf=None
        funds["operating_cf"] = pd.concat([
            funds["operating_cf"],
            _make_quarterly_df(q_dates, [30e9, 30e9, 30e9, 30e9]),
        ], ignore_index=True)
        nd = normalise(funds, _make_ohlcv(), _base_info())
        assert nd.ttm_fcf is None

    def test_annual_series_unchanged_by_ttm(self):
        nd_base = normalise(_base_fundamentals(), _make_ohlcv(), _base_info())
        nd_ttm  = normalise(self._base_with_quarters(), _make_ohlcv(), _base_info())
        # Annual revenue should be identical
        assert nd_base.revenue_annual == nd_ttm.revenue_annual
