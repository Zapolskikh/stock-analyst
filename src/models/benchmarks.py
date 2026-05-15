"""
Benchmark models — Шаг 4 алгоритма.

Для каждого CompanyType определяет:
  1. Веса пяти блоков оценки (Quality / Valuation / Technical / Risk / StyleFit).
     Веса в сумме дают 1.0.
  2. Пороговые шкалы для каждой метрики — список точек (raw_value → score 0–10)
     с линейной интерполяцией.  Направление "хуже→лучше" задаётся порядком точек:
       ascending=True  (higher is better): [(0,0), (10,5), (20,10)]
       ascending=False (lower is better):  [(50,0), (25,5), (0,10)]

Публичный интерфейс
-------------------
    from src.models.benchmarks import get_benchmark, score_metric, Benchmark

    bm = get_benchmark(CompanyType.HYPERGROWTH_TECH)
    bm.weights.quality          # 0.30
    bm.weights.valuation        # 0.20
    score = score_metric(bm, "revenue_growth", 28.0)  # → float 0–10
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.classifier import CompanyType


# ---------------------------------------------------------------------------
# Threshold: (raw_value, score) breakpoint list + linear interpolation
# ---------------------------------------------------------------------------

@dataclass
class Threshold:
    """
    Maps a raw metric value to a score in [0, 10] via linear interpolation.

    *points* is a list of (raw_value, score) pairs **sorted by raw_value**.
    Values outside the range are clamped to the boundary score.

    Example — higher revenue growth is better:
        Threshold([(0, 0), (10, 4), (20, 7), (35, 10)])

    Example — lower P/E is better:
        Threshold([(10, 10), (20, 7), (35, 4), (60, 0)])
    """
    points: list[tuple[float, float]]

    def score(self, value: float) -> float:
        """Return interpolated score for *value*, or NaN if value is NaN/inf."""
        if not math.isfinite(value):
            return float("nan")
        pts = self.points
        if not pts:
            return float("nan")
        # Clamp below
        if value <= pts[0][0]:
            return pts[0][1]
        # Clamp above
        if value >= pts[-1][0]:
            return pts[-1][1]
        # Linear interpolation
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            if x0 <= value <= x1:
                t = (value - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return float("nan")  # should never reach here


# ---------------------------------------------------------------------------
# Block weights
# ---------------------------------------------------------------------------

@dataclass
class BlockWeights:
    quality:    float   # Business Quality
    valuation:  float   # Valuation
    technical:  float   # Technical State
    risk:       float   # Risk
    style_fit:  float   # Style Fit (type-specific bonus)

    def __post_init__(self) -> None:
        total = self.quality + self.valuation + self.technical + self.risk + self.style_fit
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"BlockWeights must sum to 1.0, got {total:.4f}")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@dataclass
class Benchmark:
    company_type: CompanyType
    weights: BlockWeights
    # metric_name → Threshold
    thresholds: dict[str, Threshold] = field(default_factory=dict)

    def score_metric(self, metric: str, value: float) -> float:
        """Score a raw metric value using this benchmark's threshold table."""
        th = self.thresholds.get(metric)
        if th is None:
            return float("nan")
        return th.score(value)


# ---------------------------------------------------------------------------
# Threshold library helpers (reusable building blocks)
# ---------------------------------------------------------------------------

def _rev_growth_ascending(low: float, mid: float, high: float) -> Threshold:
    """Revenue growth threshold: negative=bad, low/mid/high breakpoints."""
    return Threshold([(-30, 0), (0, 2), (low, 5), (mid, 7.5), (high, 10)])


def _margin_threshold(poor: float, fair: float, good: float, great: float) -> Threshold:
    return Threshold([(poor, 0), (fair, 3), (good, 6), (great, 10)])


def _pe_threshold(cheap: float, fair_lo: float, fair_hi: float, expensive: float) -> Threshold:
    """Lower P/E → higher score (value investing lens)."""
    return Threshold([(cheap, 10), (fair_lo, 7), (fair_hi, 4), (expensive, 0)])


def _fcf_margin_threshold(poor: float, ok: float, good: float, great: float) -> Threshold:
    """FCF margin % — higher is better."""
    return Threshold([(poor, 0), (ok, 4), (good, 7), (great, 10)])


def _ps_threshold(cheap: float, fair: float, rich: float, very_rich: float) -> Threshold:
    """P/S ratio — lower is better (value lens). Breakpoints vary by type.

    SaaS/Hypergrowth: higher P/S acceptable (rich=15, very_rich=30).
    Retail/Cyclical:  P/S > 2 already expensive (rich=3, very_rich=6).
    """
    return Threshold([(cheap, 10), (fair, 7), (rich, 3), (very_rich, 0)])


def _fcf_yield_threshold(low: float, ok: float, good: float, great: float) -> Threshold:
    """FCF yield % — higher is better. Breakpoints vary by type.

    Growth: любой положительный FCF хорош; mature: ждут 4%+.
    """
    return Threshold([(0, 0), (low, 3), (ok, 6), (good, 8), (great, 10)])


def _peg_threshold(great: float, good: float, fair: float, expensive: float) -> Threshold:
    """PEG ratio — lower is better. PEG < 1 = рост дёшев.

    Используется только для типов, где рост является ключевым тезисом.
    Для Dividend/Cyclical/Financial — не добавляется в бенчмарк.
    """
    return Threshold([(great, 10), (good, 7), (fair, 4), (expensive, 0)])


def _de_threshold(safe: float, moderate: float, high: float, danger: float) -> Threshold:
    """Debt-to-equity — lower is better (safe→high → score 10→0)."""
    return Threshold([(safe, 10), (moderate, 7), (high, 3), (danger, 0)])


def _ev_ebitda_threshold(cheap: float, fair: float, rich: float, very_rich: float) -> Threshold:
    """EV/EBITDA — lower is better.

    cheap → 10, fair → 7, rich → 3, very_rich → 0.
    Ranges differ by type: growth tolerates higher multiples.
    """
    return Threshold([(cheap, 10), (fair, 7), (rich, 3), (very_rich, 0)])


def _roe_threshold(low: float, ok: float, good: float, great: float) -> Threshold:
    return Threshold([(low, 0), (ok, 4), (good, 7), (great, 10)])


def _beta_threshold() -> Threshold:
    """Beta 0–1 is good, >2 is risky."""
    return Threshold([(0.0, 10), (0.8, 9), (1.2, 7), (1.8, 4), (3.0, 0)])


def _dividend_yield_threshold(target_lo: float, target_hi: float) -> Threshold:
    """Dividend yield scored around a target band."""
    return Threshold([(0.0, 2), (target_lo, 7), (target_hi, 10), (target_hi * 2, 5)])


# ---------------------------------------------------------------------------
# Per-type benchmark definitions
# ---------------------------------------------------------------------------

def _hypergrowth_tech() -> Benchmark:
    return Benchmark(
        company_type=CompanyType.HYPERGROWTH_TECH,
        weights=BlockWeights(
            quality=0.30,
            valuation=0.20,
            technical=0.20,
            risk=0.15,
            style_fit=0.15,
        ),
        thresholds={
            # Quality block
            "revenue_growth":        _rev_growth_ascending(15, 25, 40),
            "eps_growth":            _rev_growth_ascending(10, 25, 50),
            "gross_margin":          _margin_threshold(30, 50, 65, 80),
            "operating_margin":      _margin_threshold(0, 10, 20, 35),
            "net_margin":            _margin_threshold(0, 8, 15, 25),
            "roe":                   _roe_threshold(0, 15, 30, 50),
            "fcf_margin":            _fcf_margin_threshold(-5, 5, 15, 25),
            # Valuation block — growth cos tolerate higher P/E
            "pe_trailing":           Threshold([(10, 10), (30, 8), (60, 5), (100, 2), (150, 0)]),
            "pe_forward":            Threshold([(10, 10), (25, 8), (50, 5), (80, 2), (120, 0)]),
            # P/S: SaaS/chips — P/S 10+ норма при высокой марже
            "ps_ratio":              _ps_threshold(2, 6, 15, 30),
            # FCF yield: любой положительный хорош для growth
            "fcf_yield":             _fcf_yield_threshold(1, 2, 4, 7),
            # PEG: главный мультипликатор для growth — P/E без учёта роста misleading
            "peg_ratio":             _peg_threshold(0.5, 1.0, 2.0, 4.0),
            # EV/EBITDA: growth tolerates high multiples
            "ev_to_ebitda":          _ev_ebitda_threshold(20, 40, 80, 150),
            "beta":                  _beta_threshold(),
            # Style fit — high growth is the core thesis
            "revenue_growth_style":  _rev_growth_ascending(20, 30, 45),
            "gross_margin_style":    _margin_threshold(40, 55, 70, 85),
        },
    )


def _mature_tech() -> Benchmark:
    return Benchmark(
        company_type=CompanyType.MATURE_TECH,
        weights=BlockWeights(
            quality=0.30,
            valuation=0.25,
            technical=0.15,
            risk=0.20,
            style_fit=0.10,
        ),
        thresholds={
            "revenue_growth":        _rev_growth_ascending(3, 8, 15),
            "eps_growth":            _rev_growth_ascending(3, 8, 15),
            "gross_margin":          _margin_threshold(30, 45, 55, 70),
            "operating_margin":      _margin_threshold(5, 15, 22, 30),
            "net_margin":            _margin_threshold(5, 12, 18, 25),
            "roe":                   _roe_threshold(5, 15, 25, 40),
            "fcf_margin":            _fcf_margin_threshold(0, 10, 18, 28),
            "pe_trailing":           _pe_threshold(12, 20, 30, 45),
            "pe_forward":            _pe_threshold(10, 17, 26, 40),
            # P/S: зрелый tech — P/S > 8 дорого
            "ps_ratio":              _ps_threshold(1, 3, 8, 15),
            # FCF yield: зрелые компании должны генерировать кэш
            "fcf_yield":             _fcf_yield_threshold(1.5, 3, 6, 10),
            # PEG: умеренный рост → PEG 1.5–2.5 справедливо
            "peg_ratio":             _peg_threshold(0.8, 1.5, 2.5, 4.0),
            # EV/EBITDA: зрелый tech — выше 25 уже дорого
            "ev_to_ebitda":          _ev_ebitda_threshold(10, 18, 30, 50),
            "debt_to_equity":        _de_threshold(0.3, 1.0, 2.5, 5.0),
            "beta":                  _beta_threshold(),
            "revenue_growth_style":  _rev_growth_ascending(4, 9, 16),
            "fcf_margin_style":      _fcf_margin_threshold(5, 15, 22, 30),
        },
    )


def _pharma() -> Benchmark:
    return Benchmark(
        company_type=CompanyType.PHARMA,
        weights=BlockWeights(
            quality=0.28,
            valuation=0.22,
            technical=0.15,
            risk=0.20,
            style_fit=0.15,
        ),
        thresholds={
            "revenue_growth":        _rev_growth_ascending(2, 7, 15),
            "eps_growth":            _rev_growth_ascending(2, 7, 15),
            "gross_margin":          _margin_threshold(40, 55, 65, 80),
            "operating_margin":      _margin_threshold(5, 15, 22, 32),
            "net_margin":            _margin_threshold(5, 12, 18, 28),
            "roe":                   _roe_threshold(5, 12, 22, 35),
            "fcf_margin":            _fcf_margin_threshold(0, 8, 16, 25),
            "pe_trailing":           _pe_threshold(12, 18, 28, 45),
            "pe_forward":            _pe_threshold(10, 15, 24, 40),
            # P/S: фарма — P/S 3–5 норма
            "ps_ratio":              _ps_threshold(1, 3, 8, 15),
            # FCF yield: pipeline-инвестиции снижают FCF
            "fcf_yield":             _fcf_yield_threshold(1, 2.5, 5, 9),
            # PEG: рост умеренный, PEG 1.5 справедливо
            "peg_ratio":             _peg_threshold(0.8, 1.5, 2.5, 4.0),
            # EV/EBITDA: pipeline premium — выше 20 норма
            "ev_to_ebitda":          _ev_ebitda_threshold(10, 18, 30, 50),
            "debt_to_equity":        _de_threshold(0.3, 1.0, 2.5, 5.0),
            "beta":                  Threshold([(0.0, 10), (0.6, 9), (1.0, 7), (1.5, 4), (2.5, 0)]),
            # Style fit — R&D intensity is a positive pharma signal (scored 0–10)
            "rd_to_revenue":         Threshold([(0, 0), (5, 3), (12, 7), (20, 10), (40, 8)]),
            "gross_margin_style":    _margin_threshold(50, 65, 75, 85),
        },
    )


def _dividend_defensive() -> Benchmark:
    return Benchmark(
        company_type=CompanyType.DIVIDEND_DEFENSIVE,
        weights=BlockWeights(
            quality=0.25,
            valuation=0.25,
            technical=0.10,
            risk=0.25,
            style_fit=0.15,
        ),
        thresholds={
            "revenue_growth":        _rev_growth_ascending(-2, 2, 6),
            "eps_growth":            _rev_growth_ascending(-2, 2, 6),
            "gross_margin":          _margin_threshold(20, 35, 48, 60),
            "operating_margin":      _margin_threshold(8, 14, 20, 28),
            "net_margin":            _margin_threshold(5, 10, 15, 22),
            "roe":                   _roe_threshold(5, 10, 18, 28),
            "fcf_margin":            _fcf_margin_threshold(3, 8, 14, 20),
            "pe_trailing":           _pe_threshold(10, 16, 24, 35),
            "pe_forward":            _pe_threshold(9, 14, 22, 32),
            # P/S: defensive — P/S > 4 уже дорого
            "ps_ratio":              _ps_threshold(0.5, 1.5, 4, 8),
            # FCF yield: дивидендный тезис требует щедрого cash flow
            "fcf_yield":             _fcf_yield_threshold(2, 4, 7, 12),
            # PEG не используется: рост не является тезисом
            # EV/EBITDA: defensive — выше 15 уже дорого
            "ev_to_ebitda":          _ev_ebitda_threshold(6, 12, 20, 35),
            "debt_to_equity":        _de_threshold(0.3, 1.0, 2.0, 4.0),
            "beta":                  Threshold([(0.0, 10), (0.5, 9), (0.8, 7), (1.2, 4), (2.0, 0)]),
            # Style fit — high dividend yield is the thesis
            "dividend_yield_pct":    _dividend_yield_threshold(2.5, 4.5),
        },
    )


def _cyclical() -> Benchmark:
    return Benchmark(
        company_type=CompanyType.CYCLICAL,
        weights=BlockWeights(
            quality=0.25,
            valuation=0.28,
            technical=0.17,
            risk=0.20,
            style_fit=0.10,
        ),
        thresholds={
            "revenue_growth":        _rev_growth_ascending(-5, 5, 15),
            "eps_growth":            _rev_growth_ascending(-10, 5, 20),
            "gross_margin":          _margin_threshold(10, 20, 30, 45),
            "operating_margin":      _margin_threshold(3, 8, 14, 22),
            "net_margin":            _margin_threshold(2, 6, 10, 16),
            "roe":                   _roe_threshold(3, 10, 18, 28),
            "fcf_margin":            _fcf_margin_threshold(-5, 4, 10, 18),
            # Cyclicals are cheap — low P/E is the entry signal
            "pe_trailing":           _pe_threshold(5, 10, 18, 30),
            "pe_forward":            _pe_threshold(4, 8, 15, 25),
            # Normalized P/E (mid-cycle): replaces trailing PE for valuation scoring.
            # Based on median of last 7yr EPS to smooth commodity / auto cycles.
            # XOM normalized ~7.8x = cheap;  GM ~5.5x = very cheap
            "normalized_pe":         _pe_threshold(4, 8, 14, 22),
            # P/S: cyclical — asset-heavy, P/S > 2 expensive
            "ps_ratio":              _ps_threshold(0.3, 1, 2.5, 5),
            # FCF yield: high yield justifies cyclical risk
            "fcf_yield":             _fcf_yield_threshold(2, 5, 8, 12),
            # EV/EBITDA: key metric for cyclicals
            "ev_to_ebitda":          _ev_ebitda_threshold(4, 8, 15, 25),
            "debt_to_equity":        _de_threshold(0.3, 1.0, 2.5, 5.0),
            "beta":                  Threshold([(0.5, 10), (1.0, 8), (1.5, 5), (2.5, 2), (3.5, 0)]),
            "revenue_growth_style":  _rev_growth_ascending(0, 8, 18),
        },
    )


def _financial() -> Benchmark:
    return Benchmark(
        company_type=CompanyType.FINANCIAL,
        weights=BlockWeights(
            quality=0.28,
            valuation=0.27,
            technical=0.13,
            risk=0.22,
            style_fit=0.10,
        ),
        thresholds={
            "revenue_growth":        _rev_growth_ascending(0, 5, 12),
            "eps_growth":            _rev_growth_ascending(0, 5, 12),
            # Financials: net margin as efficiency proxy (lower than industrials — normal)
            "net_margin":            _margin_threshold(10, 18, 25, 35),
            "roe":                   _roe_threshold(5, 10, 15, 25),
            # Low P/E typical for banks
            "pe_trailing":           _pe_threshold(6, 10, 16, 25),
            "pe_forward":            _pe_threshold(5, 9, 14, 22),
            # P/TBV — primary bank valuation metric (replaces P/S and FCF yield)
            # Below 1.0 = cheap, 1.0–2.0 = fair, > 3.0 = expensive
            "ptbv":                  Threshold([(0.5, 10), (1.0, 8), (1.5, 6), (2.5, 3), (4.0, 0)]),
            # Risk — financials carry structural leverage; D/E not comparable
            "beta":                  _beta_threshold(),
            "revenue_growth_style":  _rev_growth_ascending(2, 6, 12),
        },
    )


def _turnaround() -> Benchmark:
    return Benchmark(
        company_type=CompanyType.TURNAROUND,
        weights=BlockWeights(
            quality=0.22,
            valuation=0.28,
            technical=0.20,
            risk=0.18,
            style_fit=0.12,
        ),
        thresholds={
            # Growth trajectory matters more than absolute level
            "revenue_growth":        _rev_growth_ascending(-5, 5, 20),
            "eps_growth":            _rev_growth_ascending(-20, 0, 30),
            "gross_margin":          _margin_threshold(10, 25, 38, 55),
            "operating_margin":      _margin_threshold(-10, 0, 8, 18),
            "net_margin":            _margin_threshold(-15, 0, 6, 14),
            "roe":                   _roe_threshold(-20, 0, 10, 22),
            "fcf_margin":            _fcf_margin_threshold(-10, 0, 8, 18),
            "pe_trailing":           _pe_threshold(5, 12, 22, 40),
            "pe_forward":            _pe_threshold(5, 10, 18, 35),
            # P/S: turnaround — P/S < 1 очень дёшево, > 5 рискованно
            "ps_ratio":              _ps_threshold(0.3, 1.5, 5, 10),
            # FCF yield: пока FCF восстанавливается, ожидания ниже
            "fcf_yield":             _fcf_yield_threshold(0.5, 2, 5, 8),
            # PEG: при восстановлении EPS может расти быстро — PEG 1.5 хорошо
            "peg_ratio":             _peg_threshold(0.5, 1.5, 3.0, 6.0),
            # EV/EBITDA: turnaround — EBITDA часто негативный, skip если nan
            "ev_to_ebitda":          _ev_ebitda_threshold(5, 12, 22, 40),
            # High debt is common but is a risk
            "debt_to_equity":        _de_threshold(0.5, 1.5, 3.5, 7.0),
            "beta":                  Threshold([(0.5, 10), (1.2, 7), (2.0, 4), (3.0, 1), (4.0, 0)]),
            "revenue_growth_style":  _rev_growth_ascending(0, 10, 25),
        },
    )


def _other() -> Benchmark:
    """Fallback benchmark with equal weights and generic thresholds."""
    return Benchmark(
        company_type=CompanyType.OTHER,
        weights=BlockWeights(
            quality=0.25,
            valuation=0.25,
            technical=0.20,
            risk=0.20,
            style_fit=0.10,
        ),
        thresholds={
            "revenue_growth":    _rev_growth_ascending(0, 8, 20),
            "eps_growth":        _rev_growth_ascending(0, 8, 20),
            "gross_margin":      _margin_threshold(20, 40, 55, 70),
            "operating_margin":  _margin_threshold(5, 12, 20, 30),
            "net_margin":        _margin_threshold(3, 10, 16, 24),
            "roe":               _roe_threshold(5, 12, 22, 35),
            "fcf_margin":        _fcf_margin_threshold(0, 8, 15, 25),
            "pe_trailing":       _pe_threshold(10, 18, 30, 50),
            "pe_forward":        _pe_threshold(8, 15, 26, 45),
            "ps_ratio":          _ps_threshold(0.5, 2, 6, 12),
            "fcf_yield":         _fcf_yield_threshold(1, 3, 6, 10),
            "ev_to_ebitda":      _ev_ebitda_threshold(8, 15, 28, 50),
            "debt_to_equity":    _de_threshold(0.3, 1.0, 2.5, 5.0),
            "beta":              _beta_threshold(),
        },
    )


# ---------------------------------------------------------------------------
# Registry + public API
# ---------------------------------------------------------------------------

_REGISTRY: dict[CompanyType, Benchmark] = {
    CompanyType.HYPERGROWTH_TECH:   _hypergrowth_tech(),
    CompanyType.MATURE_TECH:        _mature_tech(),
    CompanyType.PHARMA:             _pharma(),
    CompanyType.DIVIDEND_DEFENSIVE: _dividend_defensive(),
    CompanyType.CYCLICAL:           _cyclical(),
    CompanyType.FINANCIAL:          _financial(),
    CompanyType.TURNAROUND:         _turnaround(),
    CompanyType.OTHER:              _other(),
}


def get_benchmark(company_type: CompanyType) -> Benchmark:
    """Return the Benchmark for *company_type* (falls back to OTHER if unknown)."""
    return _REGISTRY.get(company_type, _REGISTRY[CompanyType.OTHER])


def score_metric(benchmark: Benchmark, metric: str, value: float) -> float:
    """
    Convenience wrapper: score a single metric against a benchmark.

    Returns float in [0, 10], or NaN if metric not in benchmark or value is NaN.
    """
    return benchmark.score_metric(metric, value)
