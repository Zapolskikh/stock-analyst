"""
Block E — Style Fit Score (0–10).

Вопрос: насколько акция соответствует своему типу?

Логика: каждый Benchmark содержит метрики с суффиксом _style
(например, revenue_growth_style, gross_margin_style, rd_to_revenue,
dividend_yield_pct) — это те же самые метрики, но откалиброванные
под "идеальный образец" данного типа компании.

Style Fit = среднее по всем *_style метрикам из Benchmark.
"""
from __future__ import annotations

import math

from src.data.normalizer import NormalisedData
from src.models.benchmarks import Benchmark
from src.scoring.base import BlockScore, avg_scores


def _recent_mean(values: list[float], n: int = 3) -> float:
    tail = [v for v in values[-n:] if math.isfinite(v)]
    return sum(tail) / len(tail) if tail else float("nan")


def _last_valid(values: list[float]) -> float:
    for v in reversed(values):
        if math.isfinite(v):
            return v
    return float("nan")


# Map threshold key → how to extract the raw value from NormalisedData
_METRIC_EXTRACTORS: dict[str, callable] = {
    "revenue_growth_style":   lambda nd: _recent_mean(nd.revenue_growth_annual),
    "gross_margin_style":     lambda nd: _recent_mean(nd.gross_margin_annual),
    "fcf_margin_style":       lambda nd: _fcf_margin(nd),
    "rd_to_revenue":          lambda nd: _rd_ratio(nd),
    "dividend_yield_pct":     lambda nd: (nd.dividend_yield * 100.0
                                          if nd.dividend_yield is not None
                                          else float("nan")),
}


def _fcf_margin(nd: NormalisedData) -> float:
    fcf_margins = []
    for fcf, rev in zip(nd.fcf_annual, nd.revenue_annual):
        if math.isfinite(fcf) and math.isfinite(rev) and rev > 0:
            fcf_margins.append(fcf / rev * 100.0)
    return _recent_mean(fcf_margins) if fcf_margins else float("nan")


def _rd_ratio(nd: NormalisedData) -> float:
    rd  = _last_valid(nd.rd_expense_annual)
    rev = _last_valid(nd.revenue_annual)
    if math.isfinite(rd) and math.isfinite(rev) and rev > 0:
        return rd / rev * 100.0
    return float("nan")


def score_style_fit(nd: NormalisedData, bm: Benchmark) -> BlockScore:
    """
    Compute Style Fit score — how well the company matches its archetype.
    Uses only *_style and special (rd_to_revenue, dividend_yield_pct)
    thresholds from the Benchmark.
    """
    breakdown: dict[str, float] = {}
    notes: list[str] = []

    # Only evaluate metrics that end with _style or are special style metrics
    style_keys = [k for k in bm.thresholds
                  if k.endswith("_style") or k in ("rd_to_revenue", "dividend_yield_pct")]

    for key in style_keys:
        extractor = _METRIC_EXTRACTORS.get(key)
        if extractor is None:
            continue
        value = extractor(nd)
        s = bm.score_metric(key, value)
        if math.isfinite(s):
            breakdown[key] = s

    final = avg_scores(breakdown)
    if not breakdown:
        # Fall back to quality signals if no style-specific metrics exist
        final = 5.0
        notes.append("no style-specific thresholds — neutral style fit")

    return BlockScore(score=final, breakdown=breakdown, notes=notes)
