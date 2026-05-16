"""
Tests for src/scoring/* — all 5 blocks. Fully offline.
"""
from __future__ import annotations

import math

import pytest

from src.classifier import CompanyType
from src.data.normalizer import NormalisedData
from src.engine.engine import _check_stop_factors, _decision
from src.models.benchmarks import get_benchmark
from src.scoring.base import BlockScore, avg_scores
from src.scoring.quality import score_quality
from src.scoring.risk import score_risk
from src.scoring.style_fit import score_style_fit
from src.scoring.technical import (
    _drawdown_from_high,
    _momentum,
    _sma,
    _trend_quality,
    score_technical,
)
from src.scoring.valuation import score_valuation

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _nd(**kwargs) -> NormalisedData:
    base = dict(
        ticker="TEST",
        years=[2020, 2021, 2022, 2023],
        revenue_annual=[200e9, 250e9, 300e9, 350e9],
        gross_profit_annual=[100e9, 130e9, 160e9, 200e9],
        operating_income_annual=[50e9, 70e9, 90e9, 110e9],
        net_income_annual=[40e9, 55e9, 70e9, 85e9],
        operating_cf_annual=[60e9, 80e9, 100e9, 120e9],
        capex_annual=[10e9, 12e9, 14e9, 16e9],
        equity_annual=[80e9, 90e9, 100e9, 110e9],
        total_assets_annual=[200e9, 220e9, 240e9, 260e9],
        total_liabilities_annual=[120e9, 130e9, 140e9, 150e9],
        long_term_debt_annual=[50e9, 45e9, 40e9, 35e9],
        eps_diluted_annual=[5.0, 7.0, 9.0, 11.0],
        rd_expense_annual=[5e9, 6e9, 7e9, 8e9],
        fcf_annual=[50e9, 68e9, 86e9, 104e9],
        gross_margin_annual=[50.0, 52.0, 53.3, 57.1],
        operating_margin_annual=[25.0, 28.0, 30.0, 31.4],
        net_margin_annual=[20.0, 22.0, 23.3, 24.3],
        revenue_growth_annual=[float("nan"), 25.0, 20.0, 16.7],
        eps_growth_annual=[float("nan"), 40.0, 28.6, 22.2],
        roe_annual=[50.0, 61.1, 70.0, 77.3],
        roa_annual=[20.0, 25.0, 29.2, 32.7],
        debt_to_equity_annual=[0.625, 0.5, 0.4, 0.318],
        current_price=155.0,
        market_cap=2.4e12,
        pe_trailing=28.5,
        pe_forward=24.0,
        beta=1.2,
        sector="Technology",
        industry="Semiconductors",
        dividend_yield=0.002,
        years_of_history=4,
        data_quality="good",
        missing_metrics=[],
        close_prices=[100.0 + i * 0.5 for i in range(252)],  # uptrend
        spy_close_prices=[],
        shares_dilution_annual=[float("nan"), 1.0, 1.2, 0.8],
        ebitda_annual=[],
        cash_annual=[],
    )
    base.update(kwargs)
    return NormalisedData(**base)


def _bm(ct: CompanyType = CompanyType.HYPERGROWTH_TECH):
    return get_benchmark(ct)


# ---------------------------------------------------------------------------
# base.py
# ---------------------------------------------------------------------------

class TestBlockScore:
    def test_score_clamped_above_10(self):
        bs = BlockScore(score=15.0)
        assert bs.score == 10.0

    def test_score_clamped_below_0(self):
        bs = BlockScore(score=-3.0)
        assert bs.score == 0.0

    def test_nan_score_becomes_0(self):
        bs = BlockScore(score=float("nan"))
        assert bs.score == 0.0

    def test_valid_score_unchanged(self):
        bs = BlockScore(score=7.5)
        assert abs(bs.score - 7.5) < 1e-9

    def test_coverage_defaults_to_1(self):
        bs = BlockScore(score=7.0)
        assert bs.coverage == 1.0

    def test_coverage_clamped(self):
        bs = BlockScore(score=5.0, coverage=1.5)
        assert bs.coverage == 1.0
        bs2 = BlockScore(score=5.0, coverage=-0.1)
        assert bs2.coverage == 0.0


class TestAvgScores:
    def test_simple_average(self):
        assert abs(avg_scores({"a": 4.0, "b": 8.0}) - 6.0) < 1e-9

    def test_ignores_nan(self):
        assert abs(avg_scores({"a": 6.0, "b": float("nan")}) - 6.0) < 1e-9

    def test_empty_returns_zero(self):
        assert avg_scores({}) == 0.0

    def test_all_nan_returns_zero(self):
        assert avg_scores({"a": float("nan")}) == 0.0

    def test_no_penalty_when_full_coverage(self):
        # 2 available out of 2 expected → no penalty
        result = avg_scores({"a": 8.0, "b": 6.0}, expected_count=2)
        assert abs(result - 7.0) < 1e-9

    def test_penalty_when_sparse(self):
        # 1 available out of 4 expected → penalty of sqrt(1/4) = 0.5
        result = avg_scores({"a": 10.0}, expected_count=4)
        assert abs(result - 10.0 * 0.5) < 1e-9

    def test_penalty_two_of_seven(self):
        # 2 of 7 → sqrt(2/7) ≈ 0.535
        import math as _math
        result = avg_scores({"a": 10.0, "b": 10.0}, expected_count=7)
        expected = 10.0 * _math.sqrt(2 / 7)
        assert abs(result - expected) < 1e-9

    def test_no_penalty_without_expected_count(self):
        # backward-compat: no expected_count → no penalty
        result = avg_scores({"a": 10.0})
        assert abs(result - 10.0) < 1e-9

    def test_coverage_penalty_never_inflates(self):
        # penalty can only reduce, never increase
        full = avg_scores({"a": 8.0, "b": 6.0}, expected_count=2)
        partial = avg_scores({"a": 8.0}, expected_count=2)
        assert partial < full


# ---------------------------------------------------------------------------
# Block A — Quality
# ---------------------------------------------------------------------------

class TestQuality:
    def test_returns_block_score(self):
        bs = score_quality(_nd(), _bm())
        assert isinstance(bs, BlockScore)

    def test_score_in_range(self):
        bs = score_quality(_nd(), _bm())
        assert 0.0 <= bs.score <= 10.0

    def test_breakdown_has_expected_keys(self):
        bs = score_quality(_nd(), _bm())
        assert "revenue_growth" in bs.breakdown
        assert "gross_margin" in bs.breakdown

    def test_high_quality_company_scores_well(self):
        nd = _nd(
            revenue_growth_annual=[float("nan"), 35.0, 32.0, 28.0],
            gross_margin_annual=[75.0, 76.0, 77.0, 78.0],
            net_margin_annual=[25.0, 26.0, 27.0, 28.0],
            roe_annual=[40.0, 42.0, 45.0, 48.0],
        )
        bs = score_quality(nd, _bm(CompanyType.HYPERGROWTH_TECH))
        assert bs.score >= 7.0

    def test_poor_quality_company_scores_low(self):
        nd = _nd(
            revenue_growth_annual=[float("nan"), -10.0, -5.0, -8.0],
            gross_margin_annual=[10.0, 9.0, 8.0, 7.0],
            net_margin_annual=[-5.0, -3.0, -4.0, -2.0],
            roe_annual=[-10.0, -8.0, -5.0, -3.0],
        )
        bs = score_quality(nd, _bm(CompanyType.HYPERGROWTH_TECH))
        assert bs.score <= 5.5

    def test_empty_nd_returns_zero_score(self):
        bs = score_quality(NormalisedData(ticker="X"), _bm())
        assert bs.score == 0.0

    def test_notes_populated_for_strong_company(self):
        nd = _nd(
            revenue_growth_annual=[float("nan"), 35.0, 30.0, 28.0],
            roe_annual=[40.0, 42.0, 45.0, 48.0],
        )
        bs = score_quality(nd, _bm())
        assert len(bs.notes) >= 1

    def test_coverage_reported(self):
        bs = score_quality(_nd(), _bm())
        assert 0.0 < bs.coverage <= 1.0

    def test_sparse_data_scores_lower_than_full(self):
        # Full data company
        full = _nd()
        # Sparse company — only revenue_growth data, everything else NaN
        nan_list = [float("nan")] * 4
        sparse = _nd(
            eps_growth_annual=nan_list,
            gross_margin_annual=nan_list,
            operating_margin_annual=nan_list,
            net_margin_annual=nan_list,
            roe_annual=nan_list,
            fcf_annual=[float("nan")] * 4,
        )
        bm = _bm()
        assert score_quality(sparse, bm).score < score_quality(full, bm).score

    def test_full_coverage_equals_1(self):
        bs = score_quality(_nd(), _bm())
        # With complete data all 7 metrics should score → coverage ≈ 1.0
        assert bs.coverage == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Block B — Valuation
# ---------------------------------------------------------------------------

class TestValuation:
    def test_returns_block_score(self):
        bs = score_valuation(_nd(), _bm())
        assert isinstance(bs, BlockScore)

    def test_score_in_range(self):
        bs = score_valuation(_nd(), _bm())
        assert 0.0 <= bs.score <= 10.0

    def test_cheap_stock_scores_higher_than_expensive(self):
        cheap = _nd(pe_trailing=15.0, pe_forward=12.0, market_cap=0.5e12)
        expensive = _nd(pe_trailing=80.0, pe_forward=70.0, market_cap=10e12)
        bm = _bm(CompanyType.MATURE_TECH)
        assert score_valuation(cheap, bm).score > score_valuation(expensive, bm).score

    def test_no_pe_still_works(self):
        nd = _nd(pe_trailing=None, pe_forward=None)
        bs = score_valuation(nd, _bm())
        assert isinstance(bs, BlockScore)

    def test_ps_ratio_computed_from_market_cap_and_revenue(self):
        nd = _nd(market_cap=700e9, revenue_annual=[100e9, 120e9, 140e9, 160e9])
        bs = score_valuation(nd, _bm())
        assert "ps_ratio" in bs.breakdown

    def test_high_fcf_yield_boosts_score(self):
        high_fcf = _nd(fcf_annual=[10e9, 12e9, 14e9, 20e9], market_cap=200e9)
        low_fcf  = _nd(fcf_annual=[1e9,  1e9,  1e9,  1e9],  market_cap=200e9)
        bm = _bm()
        assert score_valuation(high_fcf, bm).score > score_valuation(low_fcf, bm).score

    def test_no_market_cap_still_works(self):
        nd = _nd(market_cap=None)
        bs = score_valuation(nd, _bm())
        assert isinstance(bs, BlockScore)

    def test_ps_ratio_type_specific_hypergrowth_vs_cyclical(self):
        """P/S 8 — нормально для Hypergrowth, дорого для Cyclical."""
        nd = _nd(market_cap=1.28e12, revenue_annual=[100e9, 120e9, 140e9, 160e9])
        # P/S = 1.28e12 / 160e9 = 8.0
        bm_hg  = _bm(CompanyType.HYPERGROWTH_TECH)
        bm_cyc = _bm(CompanyType.CYCLICAL)
        score_hg  = score_valuation(nd, bm_hg).breakdown.get("ps_ratio", 0)
        score_cyc = score_valuation(nd, bm_cyc).breakdown.get("ps_ratio", 0)
        assert score_hg > score_cyc, (
            f"Hypergrowth P/S score {score_hg:.1f} should be > Cyclical {score_cyc:.1f}"
        )

    def test_fcf_yield_type_specific_dividend_vs_growth(self):
        """FCF yield 2% — приемлемо для Hypergrowth, мало для Dividend."""
        nd = _nd(fcf_annual=[4e9, 5e9, 5e9, 4e9], market_cap=200e9)  # yield ~2%
        bm_hg = _bm(CompanyType.HYPERGROWTH_TECH)
        bm_dd = _bm(CompanyType.DIVIDEND_DEFENSIVE)
        score_hg = score_valuation(nd, bm_hg).breakdown.get("fcf_yield", 0)
        score_dd = score_valuation(nd, bm_dd).breakdown.get("fcf_yield", 0)
        assert score_hg > score_dd, (
            f"Hypergrowth FCF yield score {score_hg:.1f} should be > Dividend {score_dd:.1f}"
        )

    def test_peg_ratio_computed_for_growth_company(self):
        """PEG должен появляться в breakdown для прибыльной растущей компании."""
        nd = _nd(
            pe_forward=30.0,
            eps_growth_annual=[float("nan"), 25.0, 30.0, 20.0],  # avg ~25%
        )
        bm = _bm(CompanyType.HYPERGROWTH_TECH)
        bs = score_valuation(nd, bm)
        assert "peg_ratio" in bs.breakdown

    def test_peg_ratio_skipped_for_negative_eps_growth(self):
        """PEG не считается при отрицательном или нулевом росте EPS."""
        nd = _nd(
            pe_forward=20.0,
            eps_growth_annual=[float("nan"), -10.0, -5.0, -3.0],
        )
        bm = _bm(CompanyType.HYPERGROWTH_TECH)
        bs = score_valuation(nd, bm)
        assert "peg_ratio" not in bs.breakdown

    def test_peg_ratio_skipped_for_types_without_threshold(self):
        """PEG не добавляется в бенчмарки без peg_ratio шкалы (Cyclical, Dividend)."""
        nd = _nd(pe_forward=15.0, eps_growth_annual=[float("nan"), 5.0, 6.0, 7.0])
        for ct in (CompanyType.CYCLICAL, CompanyType.DIVIDEND_DEFENSIVE, CompanyType.FINANCIAL):
            bs = score_valuation(nd, _bm(ct))
            assert "peg_ratio" not in bs.breakdown, f"PEG should not be in {ct.value} breakdown"

    def test_low_peg_notes_attractive(self):
        """PEG < 1 — должна быть заметка 'attractive'."""
        nd = _nd(
            pe_forward=20.0,
            eps_growth_annual=[float("nan"), 30.0, 35.0, 25.0],  # avg ~30% → PEG ≈ 0.67
        )
        bs = score_valuation(nd, _bm(CompanyType.HYPERGROWTH_TECH))
        combined = " ".join(bs.notes).lower()
        assert "attractive" in combined or "peg" in combined


# ---------------------------------------------------------------------------
# Block C — Technical helpers
# ---------------------------------------------------------------------------

class TestTechnicalHelpers:
    def _prices(self, n=252, start=100, step=0.5):
        return [start + i * step for i in range(n)]

    def test_sma_basic(self):
        prices = [float(i) for i in range(1, 201)]
        result = _sma(prices, 50)
        assert result is not None
        assert abs(result - 175.5) < 1.0

    def test_sma_insufficient_data_returns_none(self):
        assert _sma([1.0, 2.0], 50) is None

    def test_momentum_uptrend(self):
        prices = [100.0 + i for i in range(130)]
        m = _momentum(prices, 63)
        assert m is not None and m > 0

    def test_momentum_downtrend(self):
        prices = [200.0 - i for i in range(130)]
        m = _momentum(prices, 63)
        assert m is not None and m < 0

    def test_momentum_insufficient_data(self):
        assert _momentum([1.0, 2.0], 63) is None

    def test_drawdown_no_decline_returns_zero(self):
        prices = [100.0 + i for i in range(252)]
        dd = _drawdown_from_high(prices)
        assert dd is not None and abs(dd) < 0.01  # at all-time high

    def test_drawdown_after_decline(self):
        prices = [100.0] * 100 + [150.0] * 50 + [90.0] * 102  # dropped from 150
        dd = _drawdown_from_high(prices)
        assert dd is not None and dd < -30

    def test_trend_quality_uptrend(self):
        prices = [50.0 + i * 0.3 for i in range(300)]
        tq = _trend_quality(prices)
        assert tq is not None and tq > 50

    def test_trend_quality_downtrend(self):
        prices = [200.0 - i * 0.5 for i in range(300)]
        tq = _trend_quality(prices)
        assert tq is not None and tq < 50


# ---------------------------------------------------------------------------
# Block C — Technical scoring
# ---------------------------------------------------------------------------

class TestTechnical:
    def test_returns_block_score(self):
        bs = score_technical(_nd())
        assert isinstance(bs, BlockScore)

    def test_score_in_range(self):
        bs = score_technical(_nd())
        assert 0.0 <= bs.score <= 10.0

    def test_no_price_history_returns_neutral(self):
        nd = _nd(close_prices=[])
        bs = score_technical(nd)
        assert abs(bs.score - 5.0) < 1e-9

    def test_uptrend_scores_higher_than_downtrend(self):
        up   = _nd(close_prices=[100.0 + i * 0.5 for i in range(252)])
        down = _nd(close_prices=[220.0 - i * 0.5 for i in range(252)])
        assert score_technical(up).score > score_technical(down).score

    def test_breakdown_keys_present_with_enough_data(self):
        bs = score_technical(_nd())
        assert "price_vs_ma50" in bs.breakdown
        assert "drawdown" in bs.breakdown

    def test_below_ma200_noted(self):
        # Price below MA200: start high, decline
        prices = [200.0 - i * 0.3 for i in range(252)]
        nd = _nd(close_prices=prices)
        bs = score_technical(nd)
        combined = " ".join(bs.notes).lower()
        assert "ma200" in combined or "bear" in combined


# ---------------------------------------------------------------------------
# Block D — Risk
# ---------------------------------------------------------------------------

class TestRisk:
    def test_returns_block_score(self):
        bs = score_risk(_nd(), _bm())
        assert isinstance(bs, BlockScore)

    def test_score_in_range(self):
        bs = score_risk(_nd(), _bm())
        assert 0.0 <= bs.score <= 10.0

    def test_low_risk_scores_higher(self):
        safe = _nd(
            debt_to_equity_annual=[0.1, 0.1, 0.1, 0.1],
            beta=0.6,
            net_margin_annual=[20.0, 21.0, 22.0, 23.0],   # stable
            fcf_annual=[10e9, 11e9, 12e9, 13e9],           # always positive
        )
        risky = _nd(
            debt_to_equity_annual=[4.0, 4.5, 5.0, 5.5],
            beta=2.8,
            net_margin_annual=[-5.0, 10.0, -8.0, 15.0],   # volatile
            fcf_annual=[-5e9, -3e9, 2e9, -1e9],            # mostly negative
        )
        bm = _bm(CompanyType.MATURE_TECH)
        assert score_risk(safe, bm).score > score_risk(risky, bm).score

    def test_high_de_noted(self):
        nd = _nd(debt_to_equity_annual=[5.0, 5.5, 6.0, 6.5])
        bs = score_risk(nd, _bm(CompanyType.MATURE_TECH))
        combined = " ".join(bs.notes).lower()
        assert "leverage" in combined or "d/e" in combined

    def test_financial_sector_skips_de_metric(self):
        """D/E не должен влиять на балл для Financial-компаний."""
        high_de = _nd(debt_to_equity_annual=[10.0, 12.0, 14.0, 15.0])
        bm = _bm(CompanyType.FINANCIAL)
        bs = score_risk(high_de, bm, CompanyType.FINANCIAL)
        assert "debt_to_equity" not in bs.breakdown

    def test_non_financial_includes_de_metric(self):
        """D/E должен влиять для не-Financial типов."""
        nd = _nd(debt_to_equity_annual=[1.0, 1.0, 1.0, 1.0])
        bm = _bm(CompanyType.MATURE_TECH)
        bs = score_risk(nd, bm, CompanyType.MATURE_TECH)
        assert "debt_to_equity" in bs.breakdown

    def test_earnings_stability_robust_near_zero_mean(self):
        """Turnaround: переход -5 → +5 не должен взрывать earnings_stability."""
        # mean ≈ 0, старый CV давал здесь NaN или inf
        nd = _nd(net_margin_annual=[-5.0, -2.0, 1.0, 4.0])
        bm = _bm(CompanyType.TURNAROUND)
        bs = score_risk(nd, bm, CompanyType.TURNAROUND)
        # Самое важное: не падает, score конечный
        assert math.isfinite(bs.score)
        # Если earnings_stability есть, он должен быть в [0, 10]
        if "earnings_stability" in bs.breakdown:
            assert 0.0 <= bs.breakdown["earnings_stability"] <= 10.0

    def test_earnings_stability_stable_scores_high(self):
        """Perfectly stable margin → максимальный instability score."""
        nd = _nd(net_margin_annual=[20.0, 20.0, 20.0, 20.0])
        bm = _bm(CompanyType.MATURE_TECH)
        bs = score_risk(nd, bm, CompanyType.MATURE_TECH)
        assert bs.breakdown.get("earnings_stability", 0) == pytest.approx(10.0)
        nd = _nd(net_margin_annual=[20.0, 21.0, 22.0, 23.0])
        bs = score_risk(nd, _bm(CompanyType.MATURE_TECH))
        assert "earnings_stability" in bs.breakdown

    def test_empty_nd_returns_neutral(self):
        bs = score_risk(NormalisedData(ticker="X"), _bm())
        assert abs(bs.score - 5.0) < 1e-9

    def test_all_positive_fcf_boosts_score(self):
        good_fcf = _nd(fcf_annual=[10e9, 11e9, 12e9, 13e9])
        bad_fcf  = _nd(fcf_annual=[-1e9, -2e9, -1e9, -3e9])
        bm = _bm()
        assert score_risk(good_fcf, bm).score > score_risk(bad_fcf, bm).score


# ---------------------------------------------------------------------------
# Block E — Style Fit
# ---------------------------------------------------------------------------

class TestStyleFit:
    def test_returns_block_score(self):
        bs = score_style_fit(_nd(), _bm())
        assert isinstance(bs, BlockScore)

    def test_score_in_range(self):
        bs = score_style_fit(_nd(), _bm())
        assert 0.0 <= bs.score <= 10.0

    def test_hypergrowth_company_fits_hypergrowth_benchmark(self):
        nd = _nd(
            revenue_growth_annual=[float("nan"), 35.0, 32.0, 28.0],
            gross_margin_annual=[75.0, 76.0, 77.0, 78.0],
        )
        bm_hg = _bm(CompanyType.HYPERGROWTH_TECH)
        # A hypergrowth company should score at least 6.0 on its own benchmark
        assert score_style_fit(nd, bm_hg).score >= 6.0

    def test_dividend_company_fits_dividend_benchmark(self):
        nd = _nd(dividend_yield=0.04)  # 4 %
        bm_dd = _bm(CompanyType.DIVIDEND_DEFENSIVE)
        bm_hg = _bm(CompanyType.HYPERGROWTH_TECH)
        assert score_style_fit(nd, bm_dd).score >= score_style_fit(nd, bm_hg).score

    def test_pharma_benchmark_has_rd_style(self):
        nd = _nd(
            sector="Healthcare",
            rd_expense_annual=[2e9, 2.2e9, 2.5e9, 2.8e9],
            revenue_annual=[10e9, 11e9, 12e9, 13e9],
        )
        bm = _bm(CompanyType.PHARMA)
        bs = score_style_fit(nd, bm)
        assert "rd_to_revenue" in bs.breakdown

    def test_no_style_thresholds_returns_neutral(self):
        # OTHER benchmark has no *_style metrics
        nd = _nd()
        bm = _bm(CompanyType.OTHER)
        bs = score_style_fit(nd, bm)
        assert abs(bs.score - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# Cross-block: all blocks together for one company
# ---------------------------------------------------------------------------

class TestAllBlocksTogether:
    def test_all_scores_in_range(self):
        nd = _nd()
        bm = _bm(CompanyType.HYPERGROWTH_TECH)
        for fn, args in [
            (score_quality,   (nd, bm)),
            (score_valuation, (nd, bm)),
            (score_technical, (nd,)),
            (score_risk,      (nd, bm)),
            (score_style_fit, (nd, bm)),
        ]:
            bs = fn(*args)
            assert 0.0 <= bs.score <= 10.0, f"{fn.__name__} out of range: {bs.score}"

    def test_breakdown_dicts_contain_only_finite_values(self):
        nd = _nd()
        bm = _bm(CompanyType.HYPERGROWTH_TECH)
        for fn, args in [
            (score_quality,   (nd, bm)),
            (score_valuation, (nd, bm)),
            (score_technical, (nd,)),
            (score_risk,      (nd, bm)),
            (score_style_fit, (nd, bm)),
        ]:
            bs = fn(*args)
            for k, v in bs.breakdown.items():
                assert math.isfinite(v), f"{fn.__name__}.breakdown[{k!r}] = {v}"


# ---------------------------------------------------------------------------
# EV/EBITDA
# ---------------------------------------------------------------------------

class TestEvEbitda:
    def test_ev_ebitda_present_when_data_available(self):
        """EV/EBITDA должен быть в breakdown когда есть ebitda_annual и cash_annual."""
        nd = _nd(
            ebitda_annual=[80e9, 100e9, 115e9, 130e9],
            cash_annual=[30e9, 35e9, 40e9, 45e9],
            long_term_debt_annual=[50e9, 45e9, 40e9, 35e9],
            market_cap=2.4e12,
        )
        bs = score_valuation(nd, _bm(CompanyType.MATURE_TECH))
        assert "ev_to_ebitda" in bs.breakdown

    def test_ev_ebitda_skipped_when_ebitda_zero(self):
        """Скип при EBITDA ≤ 0 (потери)."""
        nd = _nd(ebitda_annual=[-5e9, -2e9, 0, 0])
        bs = score_valuation(nd, _bm(CompanyType.MATURE_TECH))
        assert "ev_to_ebitda" not in bs.breakdown

    def test_cheap_ev_ebitda_scores_higher(self):
        """Более дешёвый EV/EBITDA даёт более высокий балл."""
        # Cyclical: cheap < 4x, expensive > 25x
        cheap = _nd(
            ebitda_annual=[200e9, 220e9, 240e9, 260e9],
            cash_annual=[30e9, 35e9, 40e9, 50e9],
            long_term_debt_annual=[10e9, 10e9, 10e9, 10e9],
            market_cap=1.0e12,  # EV/EBITDA ≈ 3.7x
        )
        expensive = _nd(
            ebitda_annual=[20e9, 22e9, 24e9, 26e9],
            cash_annual=[5e9, 5e9, 5e9, 5e9],
            long_term_debt_annual=[10e9, 10e9, 10e9, 10e9],
            market_cap=1.0e12,  # EV/EBITDA ≈ 39x
        )
        bm = _bm(CompanyType.CYCLICAL)
        s_cheap = score_valuation(cheap, bm).breakdown["ev_to_ebitda"]
        s_exp   = score_valuation(expensive, bm).breakdown["ev_to_ebitda"]
        assert s_cheap > s_exp


# ---------------------------------------------------------------------------
# Dilution risk
# ---------------------------------------------------------------------------

class TestDilutionRisk:
    def test_buyback_scores_high(self):
        """Сокращение акций (buyback) должно давать высокий балл dilution_risk."""
        nd = _nd(shares_dilution_annual=[float("nan"), -3.0, -2.5, -2.0])
        bs = score_risk(nd, _bm())
        assert "dilution_risk" in bs.breakdown
        assert bs.breakdown["dilution_risk"] >= 9.0

    def test_heavy_dilution_scores_low(self):
        """Большое разводнение должно давать низкий балл."""
        nd = _nd(shares_dilution_annual=[float("nan"), 8.0, 10.0, 12.0])
        bs = score_risk(nd, _bm())
        assert "dilution_risk" in bs.breakdown
        assert bs.breakdown["dilution_risk"] <= 3.0

    def test_dilution_note_when_heavy(self):
        nd = _nd(shares_dilution_annual=[float("nan"), 7.0, 8.0, 9.0])
        bs = score_risk(nd, _bm())
        combined = " ".join(bs.notes).lower()
        assert "dilution" in combined

    def test_no_dilution_data_skips(self):
        nd = _nd(shares_dilution_annual=[])
        bs = score_risk(nd, _bm())
        assert "dilution_risk" not in bs.breakdown


# ---------------------------------------------------------------------------
# Relative Strength vs SPY
# ---------------------------------------------------------------------------

class TestRelativeStrength:
    def test_rs_present_when_spy_prices_available(self):
        """relative_strength должен быть в breakdown при наличии SPY данных."""
        prices = [float(100 + i * 0.5) for i in range(252)]
        spy    = [float(100 + i * 0.3) for i in range(252)]  # SPY растёт медленнее
        nd = _nd(close_prices=prices, spy_close_prices=spy)
        bs = score_technical(nd)
        assert "relative_strength" in bs.breakdown

    def test_outperforming_scores_higher_than_underperforming(self):
        n = 252
        # Stock outperforms: +25% за 3m (≈63 бара)
        prices_strong = [100.0] * (n - 63) + [100.0 * (1 + i * 25 / 63 / 100) for i in range(63)]
        # Stock underperforms: flat while SPY grows
        prices_weak   = [100.0] * n
        spy = [100.0] * (n - 63) + [100.0 * (1 + i * 15 / 63 / 100) for i in range(63)]

        nd_strong = _nd(close_prices=prices_strong, spy_close_prices=spy)
        nd_weak   = _nd(close_prices=prices_weak,   spy_close_prices=spy)

        s_strong = score_technical(nd_strong).breakdown.get("relative_strength", 0)
        s_weak   = score_technical(nd_weak).breakdown.get("relative_strength", 0)
        assert s_strong > s_weak

    def test_rs_skipped_without_spy_prices(self):
        nd = _nd(spy_close_prices=[])
        bs = score_technical(nd)
        assert "relative_strength" not in bs.breakdown


# ---------------------------------------------------------------------------
# Liquidity stop factor
# ---------------------------------------------------------------------------

class TestLiquidityStopFactor:
    """Liquidity stop factors now use dollar ADV (avg_volume × current_price)."""

    def test_very_illiquid_triggers_warning(self):
        # $50 stock × 50k shares/day = $2.5M/day ADV < $5M threshold
        nd = _nd(avg_volume=50_000, current_price=50.0)
        factors = _check_stop_factors(nd, {})
        names = [f.name for f in factors]
        assert "Low Liquidity" in names

    def test_limited_liquidity_triggers_warning(self):
        # $30 stock × 300k shares/day = $9M/day ADV → $5M–$20M range
        nd = _nd(avg_volume=300_000, current_price=30.0)
        factors = _check_stop_factors(nd, {})
        names = [f.name for f in factors]
        assert "Limited Liquidity" in names

    def test_liquid_stock_no_liquidity_warning(self):
        # $50 stock × 1M shares/day = $50M/day ADV > $20M threshold
        nd = _nd(avg_volume=1_000_000, current_price=50.0)
        factors = _check_stop_factors(nd, {})
        names = [f.name for f in factors]
        assert "Low Liquidity" not in names
        assert "Limited Liquidity" not in names

    def test_no_volume_data_no_warning(self):
        nd = _nd(avg_volume=None)
        factors = _check_stop_factors(nd, {})
        names = [f.name for f in factors]
        assert "Low Liquidity" not in names

    def test_fallback_share_threshold_when_no_price(self):
        # No current_price → falls back to share-count threshold: 50k < 100k → Low Liquidity
        nd = _nd(avg_volume=50_000, current_price=None)
        factors = _check_stop_factors(nd, {})
        names = [f.name for f in factors]
        assert "Low Liquidity" in names

    def test_high_share_volume_liquid_when_cheap_stock(self):
        # $2 stock × 600k shares = $1.2M/day ADV → Low Liquidity (dollar-based)
        nd = _nd(avg_volume=600_000, current_price=2.0)
        factors = _check_stop_factors(nd, {})
        names = [f.name for f in factors]
        assert "Low Liquidity" in names


# ---------------------------------------------------------------------------
# Decision threshold 6.3
# ---------------------------------------------------------------------------

class TestDecisionThresholdFixed:
    def test_buy_at_exactly_70(self):
        """Score 70 (Good Candidate) теперь должен давать Buy, не Watch."""
        assert _decision(70, []) == "Buy"

    def test_buy_on_limit_at_69(self):
        assert _decision(69, []) == "Buy on Limit"
