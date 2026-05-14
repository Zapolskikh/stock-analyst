"""
Tests for src/engine/engine.py and src/output/formatter.py — fully offline.
"""
from __future__ import annotations

import math
import pytest

from src.classifier import CompanyType
from src.data.normalizer import NormalisedData
from src.engine.engine import (
    AnalysisResult, HorizonScores, HorizonDecisions, StopFactor,
    analyse_nd, _weighted_avg, _rating, _decision, _check_stop_factors,
)
from src.output.formatter import format_brief, format_report


# ---------------------------------------------------------------------------
# Shared helper — builds a NormalisedData with sensible defaults
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
        close_prices=[float(100 + i * 0.5) for i in range(252)],
        spy_close_prices=[],
        shares_dilution_annual=[float("nan"), 1.0, 1.2, 0.8],
        ebitda_annual=[],
        cash_annual=[],
    )
    base.update(kwargs)
    return NormalisedData(**base)


# ---------------------------------------------------------------------------
# _weighted_avg
# ---------------------------------------------------------------------------

class TestWeightedAvg:
    def test_equal_weights(self):
        scores = {"a": 8.0, "b": 6.0}
        weights = {"a": 0.5, "b": 0.5}
        assert _weighted_avg(scores, weights) == pytest.approx(7.0)

    def test_unequal_weights(self):
        scores = {"a": 10.0, "b": 0.0}
        weights = {"a": 0.8, "b": 0.2}
        assert _weighted_avg(scores, weights) == pytest.approx(8.0)

    def test_missing_key_treated_as_zero(self):
        scores = {"a": 10.0}
        weights = {"a": 0.5, "b": 0.5}
        assert _weighted_avg(scores, weights) == pytest.approx(5.0)

    def test_empty_weights_returns_zero(self):
        assert _weighted_avg({"a": 5.0}, {}) == 0.0


# ---------------------------------------------------------------------------
# _rating
# ---------------------------------------------------------------------------

class TestRating:
    def test_strong_candidate(self):
        assert _rating(90) == "Strong Candidate"

    def test_good_candidate(self):
        assert _rating(72) == "Good Candidate"

    def test_neutral(self):
        assert _rating(60) == "Neutral / Watchlist"

    def test_weak(self):
        assert _rating(45) == "Weak"

    def test_avoid(self):
        assert _rating(30) == "Avoid"

    def test_exact_boundary_85(self):
        assert _rating(85) == "Strong Candidate"

    def test_exact_boundary_70(self):
        assert _rating(70) == "Good Candidate"


# ---------------------------------------------------------------------------
# _decision
# ---------------------------------------------------------------------------

class TestDecision:
    def test_buy_high_score_no_stops(self):
        assert _decision(80, []) == "Buy"

    def test_buy_at_exactly_70(self):
        """Buy threshold aligned with Good Candidate (70)."""
        assert _decision(70, []) == "Buy"

    def test_watch_at_69(self):
        assert _decision(69, []) == "Watch"

    def test_watch_mid_score(self):
        assert _decision(60, []) == "Watch"

    def test_hold_low_score(self):
        assert _decision(45, []) == "Hold"

    def test_avoid_very_low(self):
        assert _decision(25, []) == "Avoid"

    def test_critical_stop_overrides_buy(self):
        stops = [StopFactor("X", "Y", "critical")]
        assert _decision(90, stops) == "Avoid"

    def test_warning_stop_does_not_override(self):
        stops = [StopFactor("X", "Y", "warning")]
        # score=80 should still be Buy despite warning
        assert _decision(80, stops) == "Buy"


# ---------------------------------------------------------------------------
# _check_stop_factors
# ---------------------------------------------------------------------------

class TestStopFactors:
    def test_no_flags_for_healthy_company(self):
        from src.engine.engine import _check_stop_factors
        from src.scoring.quality import score_quality
        from src.models.benchmarks import get_benchmark
        nd = _nd()
        blocks = {}
        factors = _check_stop_factors(nd, blocks)
        assert factors == []

    def test_negative_fcf_warning(self):
        from src.engine.engine import _check_stop_factors
        nd = _nd(
            fcf_annual=[-10e9, -5e9, -8e9, -3e9],
            debt_to_equity_annual=[0.5, 0.5, 0.5, 0.5],
        )
        factors = _check_stop_factors(nd, {})
        names = [f.name for f in factors]
        assert "Negative FCF" in names
        fcf_flag = next(f for f in factors if f.name == "Negative FCF")
        assert fcf_flag.severity == "warning"  # D/E < 2.0

    def test_negative_fcf_with_high_debt_is_critical(self):
        from src.engine.engine import _check_stop_factors
        nd = _nd(
            fcf_annual=[-10e9, -5e9, -8e9, -3e9],
            debt_to_equity_annual=[3.0, 3.5, 4.0, 4.5],
        )
        factors = _check_stop_factors(nd, {})
        fcf_flag = next((f for f in factors if f.name == "Negative FCF"), None)
        assert fcf_flag is not None
        assert fcf_flag.severity == "critical"

    def test_extreme_pe_triggers_warning(self):
        from src.engine.engine import _check_stop_factors
        nd = _nd(pe_trailing=150.0)
        factors = _check_stop_factors(nd, {})
        names = [f.name for f in factors]
        assert "Extreme Valuation" in names

    def test_pe_below_100_no_valuation_flag(self):
        from src.engine.engine import _check_stop_factors
        nd = _nd(pe_trailing=95.0)
        factors = _check_stop_factors(nd, {})
        names = [f.name for f in factors]
        assert "Extreme Valuation" not in names

    def test_very_high_debt_is_critical(self):
        from src.engine.engine import _check_stop_factors
        nd = _nd(debt_to_equity_annual=[5.0, 5.5, 6.0, 6.5])
        # Technology company — default threshold 4.0, should be critical
        factors = _check_stop_factors(nd, {}, CompanyType.MATURE_TECH)
        debt_flag = next((f for f in factors if f.name == "High Debt"), None)
        assert debt_flag is not None
        assert debt_flag.severity == "critical"

    def test_financial_high_debt_no_stop(self):
        """Financial-сектор: высокий D/E не должен давать critical стоп."""
        from src.engine.engine import _check_stop_factors
        nd = _nd(debt_to_equity_annual=[8.0, 9.0, 10.0, 11.0])
        factors = _check_stop_factors(nd, {}, CompanyType.FINANCIAL)
        names = [f.name for f in factors]
        assert "High Debt" not in names

    def test_turnaround_higher_de_threshold(self):
        """Turnaround: D/E=6 не должен давать critical (порог 8.0)."""
        from src.engine.engine import _check_stop_factors
        nd = _nd(debt_to_equity_annual=[5.0, 5.5, 6.0, 6.0])
        factors = _check_stop_factors(nd, {}, CompanyType.TURNAROUND)
        names = [f.name for f in factors]
        assert "High Debt" not in names

    def test_turnaround_extreme_debt_is_critical(self):
        """Turnaround: D/E=9 всё равно critical (выше порога 8.0)."""
        from src.engine.engine import _check_stop_factors
        nd = _nd(debt_to_equity_annual=[7.0, 8.0, 8.5, 9.0])
        factors = _check_stop_factors(nd, {}, CompanyType.TURNAROUND)
        debt_flag = next((f for f in factors if f.name == "High Debt"), None)
        assert debt_flag is not None
        assert debt_flag.severity == "critical"

    def test_technical_breakdown_from_low_score(self):
        from src.engine.engine import _check_stop_factors
        from src.scoring.base import BlockScore
        blocks = {"technical": BlockScore(score=2.0, breakdown={}, notes=[])}
        factors = _check_stop_factors(_nd(), blocks)
        names = [f.name for f in factors]
        assert "Technical Breakdown" in names

    def test_no_technical_flag_when_score_normal(self):
        from src.engine.engine import _check_stop_factors
        from src.scoring.base import BlockScore
        blocks = {"technical": BlockScore(score=6.0, breakdown={}, notes=[])}
        factors = _check_stop_factors(_nd(), blocks)
        names = [f.name for f in factors]
        assert "Technical Breakdown" not in names

    def test_deteriorating_margins_flag(self):
        from src.engine.engine import _check_stop_factors
        nd = _nd(net_margin_annual=[5.0, -2.0, -5.0, -15.0])
        factors = _check_stop_factors(nd, {})
        names = [f.name for f in factors]
        assert "Deteriorating Margins" in names


# ---------------------------------------------------------------------------
# analyse_nd — integration
# ---------------------------------------------------------------------------

class TestAnalyseNd:
    def test_returns_analysis_result(self):
        result = analyse_nd(_nd())
        assert isinstance(result, AnalysisResult)

    def test_overall_score_in_range(self):
        result = analyse_nd(_nd())
        assert 0.0 <= result.overall_score <= 100.0

    def test_horizon_scores_in_range(self):
        result = analyse_nd(_nd())
        for s in (result.horizon.short, result.horizon.medium, result.horizon.long):
            assert 0.0 <= s <= 100.0

    def test_ticker_preserved(self):
        result = analyse_nd(_nd(ticker="AAPL"))
        assert result.ticker == "AAPL"

    def test_block_scores_present(self):
        result = analyse_nd(_nd())
        for key in ("quality", "valuation", "technical", "risk", "style_fit"):
            assert key in result.block_scores

    def test_rating_is_valid_string(self):
        result = analyse_nd(_nd())
        valid = {"Strong Candidate", "Good Candidate", "Neutral / Watchlist", "Weak", "Avoid"}
        assert result.rating in valid

    def test_decision_is_valid_string(self):
        result = analyse_nd(_nd())
        valid = {"Buy", "Watch", "Hold", "Avoid"}
        assert result.decision in valid

    def test_strong_company_scores_higher_than_weak(self):
        strong = analyse_nd(_nd())  # healthy defaults
        weak = analyse_nd(_nd(
            revenue_growth_annual=[float("nan"), -10.0, -15.0, -20.0],
            net_margin_annual=[-10.0, -12.0, -15.0, -18.0],
            fcf_annual=[-5e9, -8e9, -10e9, -12e9],
            debt_to_equity_annual=[3.0, 3.5, 4.0, 4.5],
            close_prices=[float(200 - i * 0.5) for i in range(252)],
        ))
        assert strong.overall_score > weak.overall_score

    def test_empty_nd_does_not_crash(self):
        result = analyse_nd(NormalisedData(ticker="EMPTY"))
        assert isinstance(result, AnalysisResult)
        assert result.overall_score >= 0.0

    def test_stop_factors_is_list(self):
        result = analyse_nd(_nd())
        assert isinstance(result.stop_factors, list)

    def test_company_type_is_enum(self):
        result = analyse_nd(_nd())
        assert isinstance(result.company_type, CompanyType)

    def test_classification_confidence_in_range(self):
        result = analyse_nd(_nd())
        assert 0.0 <= result.classification_confidence <= 1.0

    def test_data_confidence_good_by_default(self):
        result = analyse_nd(_nd(data_quality="good"))
        assert result.data_confidence == "good"

    def test_data_confidence_partial_propagated(self):
        result = analyse_nd(_nd(data_quality="partial"))
        assert result.data_confidence == "partial"

    def test_data_confidence_poor_propagated(self):
        result = analyse_nd(_nd(data_quality="poor"))
        assert result.data_confidence == "poor"

    def test_critical_stop_leads_to_avoid(self):
        # Very high debt for non-Financial company (Technology) — critical stop → Avoid
        result = analyse_nd(_nd(
            debt_to_equity_annual=[5.0, 6.0, 7.0, 8.0],
            sector="Technology",
        ))
        assert result.decision == "Avoid"

    def test_financial_high_debt_does_not_force_avoid(self):
        """Financial-сектор: высокий D/E не должен давать Avoid."""
        # Классификатор распознаёт FINANCIAL по: сектор(+50) + D/E>3(+20) = 70
        # Специально устанавливаем реалистичные банковские данные:
        # - структурный высокий леверидж (D/E > 3 через long_term_debt)
        # - P/E > 35 (убирает бонус +10 у MATURE_TECH)
        result = analyse_nd(_nd(
            sector="Financials",
            industry="Banks",
            revenue_growth_annual=[float("nan"), 5.0, 6.0, 7.0],
            eps_growth_annual=[float("nan"), 6.0, 7.0, 8.0],
            gross_margin_annual=[],
            gross_profit_annual=[],
            dividend_yield=0.025,
            # Высокий структурный леверидж → классификатор добавит +20 для Financial
            long_term_debt_annual=[400e9, 430e9, 460e9, 500e9],  # D/E ≈ 4.5x
            # D/E в стоп-факторе тоже высокий — именно это мы и тестируем
            debt_to_equity_annual=[10.0, 11.0, 12.0, 13.0],
            pe_trailing=40.0,   # > 35 → убирает +10 у MATURE_TECH
        ))
        assert result.company_type.value == "Financial", (
            f"Expected Financial, got {result.company_type.value}. "
            f"Scores: {result.classification_confidence}"
        )
        stop_names = [sf.name for sf in result.stop_factors]
        assert "High Debt" not in stop_names


# ---------------------------------------------------------------------------
# format_brief
# ---------------------------------------------------------------------------

class TestFormatBrief:
    def test_returns_string(self):
        result = analyse_nd(_nd())
        assert isinstance(format_brief(result), str)

    def test_contains_ticker(self):
        result = analyse_nd(_nd(ticker="MSFT"))
        assert "MSFT" in format_brief(result)

    def test_contains_decision(self):
        result = analyse_nd(_nd())
        brief = format_brief(result)
        assert any(d in brief for d in ("BUY", "WATCH", "HOLD", "AVOID"))

    def test_stop_factor_warning_marker(self):
        # Force a stop factor
        result = analyse_nd(_nd(debt_to_equity_annual=[5.0, 6.0, 7.0, 8.0]))
        brief = format_brief(result)
        assert "⚠" in brief

    def test_no_warning_marker_when_clean(self):
        result = analyse_nd(_nd())  # healthy company — no stops
        brief = format_brief(result)
        assert "⚠" not in brief

    def test_poor_data_quality_shows_red_in_brief(self):
        result = analyse_nd(_nd(data_quality="poor"))
        brief = format_brief(result)
        assert "🔴" in brief

    def test_partial_data_quality_shows_yellow_in_brief(self):
        result = analyse_nd(_nd(data_quality="partial"))
        brief = format_brief(result)
        assert "🟡" in brief


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_returns_string(self):
        result = analyse_nd(_nd())
        assert isinstance(format_report(result), str)

    def test_contains_ticker(self):
        result = analyse_nd(_nd(ticker="NVDA"))
        report = format_report(result)
        assert "NVDA" in report

    def test_contains_all_block_names(self):
        result = analyse_nd(_nd())
        report = format_report(result)
        for block in ("quality", "valuation", "technical", "risk", "style_fit"):
            assert block in report

    def test_contains_horizons(self):
        result = analyse_nd(_nd())
        report = format_report(result)
        assert "Short" in report and "Long" in report

    def test_stop_factors_section_when_present(self):
        result = analyse_nd(_nd(debt_to_equity_annual=[5.0, 6.0, 7.0, 8.0]))
        report = format_report(result)
        assert "STOP FACTORS" in report

    def test_no_stop_factors_section_when_clean(self):
        result = analyse_nd(_nd())
        report = format_report(result)
        assert "STOP FACTORS" not in report

    def test_contains_decision(self):
        result = analyse_nd(_nd())
        report = format_report(result)
        # Formatter renders decision as "✅ BUY", "👁  WATCH" etc.
        assert result.decision.upper() in report

    def test_contains_rating(self):
        result = analyse_nd(_nd())
        report = format_report(result)
        assert result.rating in report

    def test_poor_data_quality_shows_warning_in_report(self):
        result = analyse_nd(_nd(data_quality="poor"))
        report = format_report(result)
        assert "DATA QUALITY" in report and "POOR" in report

    def test_partial_data_quality_shows_warning_in_report(self):
        result = analyse_nd(_nd(data_quality="partial"))
        report = format_report(result)
        assert "DATA QUALITY" in report and "PARTIAL" in report

    def test_good_data_quality_no_warning_in_report(self):
        result = analyse_nd(_nd(data_quality="good"))
        report = format_report(result)
        assert "DATA QUALITY" not in report


# ---------------------------------------------------------------------------
# HorizonDecisions
# ---------------------------------------------------------------------------

class TestHorizonDecisions:

    def test_result_has_horizon_decisions(self):
        result = analyse_nd(_nd())
        assert hasattr(result, "horizon_decisions")
        assert isinstance(result.horizon_decisions, HorizonDecisions)

    def test_horizon_decisions_fields(self):
        result = analyse_nd(_nd())
        hd = result.horizon_decisions
        assert hasattr(hd, "short")
        assert hasattr(hd, "medium")
        assert hasattr(hd, "long")

    def test_all_decisions_are_valid_strings(self):
        result = analyse_nd(_nd())
        valid = {"Buy", "Watch", "Hold", "Avoid"}
        hd = result.horizon_decisions
        assert hd.short in valid, f"unexpected short decision: {hd.short!r}"
        assert hd.medium in valid, f"unexpected medium decision: {hd.medium!r}"
        assert hd.long in valid, f"unexpected long decision: {hd.long!r}"

    def test_decision_alias_equals_medium(self):
        """result.decision is a backward-compat alias for medium-horizon decision."""
        result = analyse_nd(_nd())
        assert result.decision == result.horizon_decisions.medium

    def test_critical_stop_forces_avoid_all_horizons(self):
        # D/E > 4.0 triggers critical stop for non-Financial
        result = analyse_nd(_nd(debt_to_equity_annual=[5.0, 6.0, 7.0, 8.0]))
        hd = result.horizon_decisions
        assert hd.short == "Avoid"
        assert hd.medium == "Avoid"
        assert hd.long == "Avoid"

    def test_high_score_stock_gets_buy_on_at_least_one_horizon(self):
        result = analyse_nd(_nd())
        hd = result.horizon_decisions
        decisions = {hd.short, hd.medium, hd.long}
        # With strong fundamentals, at least one horizon should be Buy or Watch
        assert decisions & {"Buy", "Watch"}, f"All horizons weak: {hd}"

    def test_short_and_long_can_differ(self):
        """Short-term uses more technical weight; long-term uses more fundamental weight.
        For the same stock the decisions can legitimately differ."""
        # This is an existence check � just verify both compute independently
        result = analyse_nd(_nd())
        # They CAN be the same � just verify they are valid and accessible
        assert result.horizon_decisions.short in {"Buy", "Watch", "Hold", "Avoid"}
        assert result.horizon_decisions.long in {"Buy", "Watch", "Hold", "Avoid"}

    def test_format_report_shows_horizon_decisions(self):
        result = analyse_nd(_nd())
        report = format_report(result)
        hd = result.horizon_decisions
        # Each horizon decision should appear in the report
        assert hd.short.lower() in report.lower() or hd.short in report
        assert hd.medium.lower() in report.lower() or hd.medium in report
        assert hd.long.lower() in report.lower() or hd.long in report

    def test_format_report_labels_horizon_decisions_section(self):
        result = analyse_nd(_nd())
        report = format_report(result)
        assert "HORIZON" in report.upper()
