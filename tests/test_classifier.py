"""
Tests for src/classifier.py — fully offline.
"""
from __future__ import annotations

from src.classifier import (
    ClassificationResult,
    CompanyType,
    _count_positive_growth,
    _last_valid,
    _recent_mean,
    classify,
)
from src.data.normalizer import NormalisedData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nd(**kwargs) -> NormalisedData:
    """Build a minimal NormalisedData with sensible defaults, overridden by kwargs."""
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
    )
    base.update(kwargs)
    return NormalisedData(**base)


# ---------------------------------------------------------------------------
# Unit tests — helpers
# ---------------------------------------------------------------------------

class TestRecentMean:
    def test_basic(self):
        assert abs(_recent_mean([1.0, 2.0, 3.0], n=3) - 2.0) < 1e-9

    def test_ignores_nan(self):
        result = _recent_mean([float("nan"), 10.0, 20.0], n=3)
        assert abs(result - 15.0) < 1e-9

    def test_empty_returns_none(self):
        assert _recent_mean([], n=3) is None

    def test_all_nan_returns_none(self):
        assert _recent_mean([float("nan"), float("nan")], n=3) is None

    def test_uses_last_n(self):
        result = _recent_mean([100.0, 1.0, 2.0, 3.0], n=3)
        assert abs(result - 2.0) < 1e-9


class TestLastValid:
    def test_returns_last_finite(self):
        assert _last_valid([1.0, 2.0, 3.0]) == 3.0

    def test_skips_trailing_nan(self):
        assert _last_valid([1.0, 2.0, float("nan")]) == 2.0

    def test_all_nan_returns_none(self):
        assert _last_valid([float("nan"), float("nan")]) is None

    def test_empty_returns_none(self):
        assert _last_valid([]) is None


class TestCountPositiveGrowth:
    def test_all_positive(self):
        assert _count_positive_growth([float("nan"), 10.0, 20.0, 5.0]) == 3

    def test_skips_first_nan(self):
        assert _count_positive_growth([float("nan"), -5.0, 10.0]) == 1

    def test_zero_not_counted(self):
        assert _count_positive_growth([float("nan"), 0.0, 5.0]) == 1


# ---------------------------------------------------------------------------
# classify() — result structure
# ---------------------------------------------------------------------------

class TestClassifyResultStructure:
    def test_returns_classification_result(self):
        result = classify(_nd())
        assert isinstance(result, ClassificationResult)

    def test_company_type_is_enum(self):
        result = classify(_nd())
        assert isinstance(result.company_type, CompanyType)

    def test_confidence_in_range(self):
        result = classify(_nd())
        assert 0.0 <= result.confidence <= 1.0

    def test_scores_contains_all_types(self):
        result = classify(_nd())
        for ct in CompanyType:
            assert ct in result.scores

    def test_signals_is_list(self):
        result = classify(_nd())
        assert isinstance(result.signals, list)

    def test_winner_has_highest_score(self):
        result = classify(_nd())
        if result.company_type != CompanyType.OTHER:
            winner_score = result.scores[result.company_type]
            for ct, s in result.scores.items():
                if ct != CompanyType.OTHER:
                    assert winner_score >= s - 1e-6


# ---------------------------------------------------------------------------
# classify() — Hypergrowth Tech
# ---------------------------------------------------------------------------

class TestHypergrowthTech:
    def _hypergrowth_nd(self):
        return _nd(
            sector="Technology",
            industry="Software",
            revenue_growth_annual=[float("nan"), 35.0, 40.0, 32.0],
            gross_margin_annual=[70.0, 72.0, 74.0, 75.0],
            dividend_yield=0.0,
            pe_trailing=55.0,
        )

    def test_classified_as_hypergrowth(self):
        result = classify(self._hypergrowth_nd())
        assert result.company_type == CompanyType.HYPERGROWTH_TECH

    def test_confidence_high(self):
        result = classify(self._hypergrowth_nd())
        assert result.confidence >= 0.5

    def test_signals_mention_growth(self):
        result = classify(self._hypergrowth_nd())
        combined = " ".join(result.signals).lower()
        assert "growth" in combined or "margin" in combined


# ---------------------------------------------------------------------------
# classify() — Mature Tech
# ---------------------------------------------------------------------------

class TestMatureTech:
    def _mature_nd(self):
        return _nd(
            sector="Technology",
            revenue_growth_annual=[float("nan"), 8.0, 10.0, 7.0],
            gross_margin_annual=[55.0, 56.0, 57.0, 58.0],
            fcf_annual=[40e9, 45e9, 50e9, 55e9],
            pe_trailing=22.0,
            dividend_yield=0.008,
        )

    def test_classified_as_mature_tech(self):
        result = classify(self._mature_nd())
        assert result.company_type == CompanyType.MATURE_TECH

    def test_not_hypergrowth(self):
        result = classify(self._mature_nd())
        assert result.company_type != CompanyType.HYPERGROWTH_TECH


# ---------------------------------------------------------------------------
# classify() — Pharma
# ---------------------------------------------------------------------------

class TestPharma:
    def _pharma_nd(self):
        rev = [10e9, 11e9, 12e9, 13e9]
        return _nd(
            sector="Healthcare",
            industry="Pharmaceuticals",
            revenue_annual=rev,
            rd_expense_annual=[2.5e9, 2.8e9, 3.0e9, 3.2e9],  # ~25% R&D ratio
            gross_margin_annual=[70.0, 71.0, 72.0, 73.0],
            revenue_growth_annual=[float("nan"), 10.0, 9.0, 8.0],
            dividend_yield=0.03,
        )

    def test_classified_as_pharma(self):
        result = classify(self._pharma_nd())
        assert result.company_type == CompanyType.PHARMA

    def test_signals_mention_sector_or_rd(self):
        result = classify(self._pharma_nd())
        combined = " ".join(result.signals).lower()
        assert "sector" in combined or "r&d" in combined or "health" in combined


# ---------------------------------------------------------------------------
# classify() — Dividend / Defensive
# ---------------------------------------------------------------------------

class TestDividendDefensive:
    def _dividend_nd(self):
        return _nd(
            sector="Utilities",
            industry="Electric Utilities",
            revenue_growth_annual=[float("nan"), 2.0, 1.5, 3.0],
            gross_margin_annual=[35.0, 36.0, 35.5, 36.5],
            net_income_annual=[5e9, 5.2e9, 5.4e9, 5.6e9],
            dividend_yield=0.04,
            pe_trailing=18.0,
        )

    def test_classified_as_dividend_defensive(self):
        result = classify(self._dividend_nd())
        assert result.company_type == CompanyType.DIVIDEND_DEFENSIVE

    def test_signals_mention_dividend(self):
        result = classify(self._dividend_nd())
        combined = " ".join(result.signals).lower()
        assert "dividend" in combined or "sector" in combined


# ---------------------------------------------------------------------------
# classify() — Cyclical
# ---------------------------------------------------------------------------

class TestCyclical:
    def _cyclical_nd(self):
        return _nd(
            sector="Energy",
            industry="Oil & Gas",
            revenue_growth_annual=[float("nan"), 40.0, -25.0, 35.0],
            net_margin_annual=[8.0, 12.0, 3.0, 9.0],
            dividend_yield=0.02,
            pe_trailing=12.0,
        )

    def test_classified_as_cyclical(self):
        result = classify(self._cyclical_nd())
        assert result.company_type == CompanyType.CYCLICAL

    def test_signals_mention_sector_or_volatility(self):
        result = classify(self._cyclical_nd())
        combined = " ".join(result.signals).lower()
        assert "sector" in combined or "cyclical" in combined or "spread" in combined


# ---------------------------------------------------------------------------
# classify() — Financial
# ---------------------------------------------------------------------------

class TestFinancial:
    def _financial_nd(self):
        return _nd(
            sector="Financial Services",
            industry="Banks",
            long_term_debt_annual=[400e9, 420e9, 430e9, 440e9],
            equity_annual=[80e9, 85e9, 88e9, 90e9],
            debt_to_equity_annual=[5.0, 4.9, 4.9, 4.9],
        )

    def test_classified_as_financial(self):
        result = classify(self._financial_nd())
        assert result.company_type == CompanyType.FINANCIAL

    def test_confidence_above_threshold(self):
        result = classify(self._financial_nd())
        assert result.confidence >= 0.30


# ---------------------------------------------------------------------------
# classify() — Turnaround
# ---------------------------------------------------------------------------

class TestTurnaround:
    def _turnaround_nd(self):
        return _nd(
            sector="Consumer Cyclical",
            net_income_annual=[-5e9, -3e9, -1e9, 2e9],
            revenue_growth_annual=[float("nan"), -20.0, -5.0, 18.0],
            long_term_debt_annual=[60e9, 65e9, 62e9, 58e9],
            equity_annual=[20e9, 18e9, 19e9, 22e9],
            debt_to_equity_annual=[3.0, 3.6, 3.3, 2.6],
        )

    def test_classified_as_turnaround(self):
        result = classify(self._turnaround_nd())
        assert result.company_type == CompanyType.TURNAROUND

    def test_signals_mention_negative_income(self):
        result = classify(self._turnaround_nd())
        combined = " ".join(result.signals).lower()
        assert "negative" in combined or "income" in combined or "recover" in combined

    def test_tech_company_with_losses_is_turnaround_not_mature(self):
        """
        Компания в Tech-секторе с убытком и падением выручки —
        sector hint НЕ должен делать её Mature Tech.
        """
        nd = _nd(
            sector="Technology",
            industry="Software",
            net_income_annual=[-8e9, -5e9, -3e9, -1e9],     # убыток все годы
            net_margin_annual=[-15.0, -10.0, -7.0, -3.0],   # убыточная маржа
            revenue_growth_annual=[float("nan"), -15.0, -8.0, 5.0],  # было падение
            gross_margin_annual=[45.0, 44.0, 43.0, 42.0],
            fcf_annual=[-3e9, -2e9, -1e9, 0.5e9],            # почти весь FCF отриц.
            pe_trailing=None,                                 # убыток → нет P/E
            dividend_yield=0.0,
        )
        result = classify(nd)
        assert result.company_type == CompanyType.TURNAROUND, (
            f"Expected Turnaround, got {result.company_type.value}. "
            f"Scores: { {k.value: round(v,1) for k,v in result.scores.items()} }"
        )


# ---------------------------------------------------------------------------
# classify() — OTHER / low confidence
# ---------------------------------------------------------------------------

class TestOther:
    def test_empty_nd_returns_other_or_low_confidence(self):
        nd = NormalisedData(ticker="EMPTY")
        result = classify(nd)
        # Either OTHER or very low confidence
        assert result.company_type == CompanyType.OTHER or result.confidence < 0.30

    def test_other_confidence_is_zero(self):
        nd = NormalisedData(ticker="EMPTY")
        result = classify(nd)
        if result.company_type == CompanyType.OTHER:
            assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Score reproducibility
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self):
        nd = _nd()
        r1 = classify(nd)
        r2 = classify(nd)
        assert r1.company_type == r2.company_type
        assert r1.confidence == r2.confidence
        assert r1.scores == r2.scores
