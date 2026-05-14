"""
Tests for src/models/benchmarks.py — fully offline.
"""
from __future__ import annotations

import math
import pytest

from src.classifier import CompanyType
from src.models.benchmarks import (
    Benchmark,
    BlockWeights,
    Threshold,
    get_benchmark,
    score_metric,
)


# ---------------------------------------------------------------------------
# Threshold interpolation
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_exact_first_point(self):
        t = Threshold([(0, 0), (10, 5), (20, 10)])
        assert t.score(0) == 0

    def test_exact_last_point(self):
        t = Threshold([(0, 0), (10, 5), (20, 10)])
        assert t.score(20) == 10

    def test_exact_middle_point(self):
        t = Threshold([(0, 0), (10, 5), (20, 10)])
        assert t.score(10) == 5

    def test_linear_interpolation_midpoint(self):
        t = Threshold([(0, 0), (10, 10)])
        assert abs(t.score(5) - 5.0) < 1e-9

    def test_linear_interpolation_quarter(self):
        t = Threshold([(0, 0), (20, 10)])
        assert abs(t.score(5) - 2.5) < 1e-9

    def test_clamp_below_minimum(self):
        t = Threshold([(0, 0), (10, 10)])
        assert t.score(-100) == 0

    def test_clamp_above_maximum(self):
        t = Threshold([(0, 0), (10, 10)])
        assert t.score(999) == 10

    def test_nan_input_returns_nan(self):
        t = Threshold([(0, 0), (10, 10)])
        assert math.isnan(t.score(float("nan")))

    def test_inf_input_returns_nan(self):
        t = Threshold([(0, 0), (10, 10)])
        assert math.isnan(t.score(float("inf")))

    def test_descending_scores(self):
        """Lower value → higher score (e.g. P/E)."""
        t = Threshold([(10, 10), (30, 5), (60, 0)])
        assert t.score(10) == 10
        assert t.score(60) == 0
        assert abs(t.score(20) - 7.5) < 1e-9

    def test_score_stays_in_0_10(self):
        t = Threshold([(0, 0), (10, 5), (20, 10)])
        for v in [-50, 0, 5, 10, 15, 20, 100]:
            s = t.score(v)
            assert 0 <= s <= 10, f"score {s} out of range for value {v}"

    def test_single_point_returns_that_score(self):
        t = Threshold([(5, 7)])
        assert t.score(0) == 7
        assert t.score(5) == 7
        assert t.score(100) == 7

    def test_three_segment_interpolation(self):
        # Points: 0→0, 10→5, 30→10
        t = Threshold([(0, 0), (10, 5), (30, 10)])
        # In segment [10, 30]: at 20 → 5 + (20-10)/(30-10) * 5 = 5 + 2.5 = 7.5
        assert abs(t.score(20) - 7.5) < 1e-9


# ---------------------------------------------------------------------------
# BlockWeights validation
# ---------------------------------------------------------------------------

class TestBlockWeights:
    def test_valid_weights_sum_to_one(self):
        bw = BlockWeights(quality=0.30, valuation=0.25, technical=0.15, risk=0.20, style_fit=0.10)
        assert math.isclose(bw.quality + bw.valuation + bw.technical + bw.risk + bw.style_fit, 1.0)

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            BlockWeights(quality=0.30, valuation=0.30, technical=0.30, risk=0.30, style_fit=0.30)


# ---------------------------------------------------------------------------
# get_benchmark — all types are registered
# ---------------------------------------------------------------------------

class TestGetBenchmark:
    @pytest.mark.parametrize("ct", list(CompanyType))
    def test_returns_benchmark_for_every_type(self, ct: CompanyType):
        bm = get_benchmark(ct)
        assert isinstance(bm, Benchmark)

    @pytest.mark.parametrize("ct", list(CompanyType))
    def test_benchmark_has_correct_type(self, ct: CompanyType):
        bm = get_benchmark(ct)
        assert bm.company_type == ct

    @pytest.mark.parametrize("ct", list(CompanyType))
    def test_weights_sum_to_one(self, ct: CompanyType):
        bm = get_benchmark(ct)
        w = bm.weights
        total = w.quality + w.valuation + w.technical + w.risk + w.style_fit
        assert math.isclose(total, 1.0, abs_tol=1e-6), f"{ct}: weights sum to {total}"

    @pytest.mark.parametrize("ct", list(CompanyType))
    def test_has_thresholds(self, ct: CompanyType):
        bm = get_benchmark(ct)
        assert len(bm.thresholds) > 0

    @pytest.mark.parametrize("ct", list(CompanyType))
    def test_core_metrics_present(self, ct: CompanyType):
        bm = get_benchmark(ct)
        # Every benchmark must have at least these core metrics
        for metric in ("revenue_growth", "net_margin", "pe_trailing"):
            assert metric in bm.thresholds, f"{ct} missing '{metric}'"


# ---------------------------------------------------------------------------
# score_metric convenience wrapper
# ---------------------------------------------------------------------------

class TestScoreMetric:
    def test_returns_float(self):
        bm = get_benchmark(CompanyType.MATURE_TECH)
        result = score_metric(bm, "revenue_growth", 10.0)
        assert isinstance(result, float)

    def test_unknown_metric_returns_nan(self):
        bm = get_benchmark(CompanyType.MATURE_TECH)
        assert math.isnan(score_metric(bm, "nonexistent_metric", 5.0))

    def test_nan_value_returns_nan(self):
        bm = get_benchmark(CompanyType.MATURE_TECH)
        assert math.isnan(score_metric(bm, "revenue_growth", float("nan")))

    def test_score_in_range(self):
        bm = get_benchmark(CompanyType.HYPERGROWTH_TECH)
        for v in [0, 10, 20, 30, 50]:
            s = score_metric(bm, "revenue_growth", float(v))
            if not math.isnan(s):
                assert 0 <= s <= 10


# ---------------------------------------------------------------------------
# Benchmark semantics — type-specific threshold ordering
# ---------------------------------------------------------------------------

class TestBenchmarkSemantics:
    """
    Verify that threshold curves encode the correct economic logic:
    e.g. Hypergrowth Tech values higher growth more than Mature Tech.
    """

    def test_hypergrowth_rewards_high_growth_more_than_mature(self):
        bm_hg = get_benchmark(CompanyType.HYPERGROWTH_TECH)
        bm_mt = get_benchmark(CompanyType.MATURE_TECH)
        # At moderate growth (10%), Mature Tech rewards it more than Hypergrowth
        # because 10% is above-average for MT but below-par for HG (expects 25-40%)
        moderate_growth = 10.0
        assert score_metric(bm_mt, "revenue_growth", moderate_growth) > \
               score_metric(bm_hg, "revenue_growth", moderate_growth)

    def test_mature_tech_rewards_moderate_growth_well(self):
        bm_mt = get_benchmark(CompanyType.MATURE_TECH)
        # 10% growth should score reasonably well for mature tech
        s = score_metric(bm_mt, "revenue_growth", 10.0)
        assert s >= 5.0

    def test_dividend_defensive_rewards_high_yield(self):
        bm = get_benchmark(CompanyType.DIVIDEND_DEFENSIVE)
        assert "dividend_yield_pct" in bm.thresholds
        high_yield = score_metric(bm, "dividend_yield_pct", 4.0)
        low_yield  = score_metric(bm, "dividend_yield_pct", 0.5)
        assert high_yield > low_yield

    def test_cyclical_tolerates_lower_margins(self):
        bm_cy = get_benchmark(CompanyType.CYCLICAL)
        bm_hg = get_benchmark(CompanyType.HYPERGROWTH_TECH)
        # A 10% net margin scores better for Cyclical than Hypergrowth
        assert score_metric(bm_cy, "net_margin", 10.0) >= \
               score_metric(bm_hg, "net_margin", 10.0)

    def test_pharma_has_rd_to_revenue_threshold(self):
        bm = get_benchmark(CompanyType.PHARMA)
        assert "rd_to_revenue" in bm.thresholds
        # High R&D intensity (~20%) should score well for pharma
        assert score_metric(bm, "rd_to_revenue", 20.0) >= 7.0

    def test_financial_pe_threshold_is_lower_range(self):
        bm_fin = get_benchmark(CompanyType.FINANCIAL)
        bm_hg  = get_benchmark(CompanyType.HYPERGROWTH_TECH)
        # At high P/E (20), Hypergrowth tolerates it better than Financials
        # because banks at P/E=20 are expensive; growth stocks at P/E=20 are cheap
        high_pe = 20.0
        assert score_metric(bm_hg, "pe_trailing", high_pe) > \
               score_metric(bm_fin, "pe_trailing", high_pe)

    def test_turnaround_tolerates_negative_margins(self):
        bm = get_benchmark(CompanyType.TURNAROUND)
        # Negative but recovering margin should not give score=0
        s = score_metric(bm, "net_margin", -5.0)
        assert s > 0

    def test_high_debt_scores_lower_than_low_debt(self):
        for ct in CompanyType:
            bm = get_benchmark(ct)
            if "debt_to_equity" not in bm.thresholds:
                continue
            low_de  = score_metric(bm, "debt_to_equity", 0.2)
            high_de = score_metric(bm, "debt_to_equity", 5.0)
            assert low_de >= high_de, f"{ct}: high D/E should not score better than low D/E"

    def test_high_beta_scores_lower_than_low_beta(self):
        for ct in CompanyType:
            bm = get_benchmark(ct)
            low_beta  = score_metric(bm, "beta", 0.5)
            high_beta = score_metric(bm, "beta", 2.5)
            if not math.isnan(low_beta) and not math.isnan(high_beta):
                assert low_beta >= high_beta, f"{ct}: high beta should not score better"
