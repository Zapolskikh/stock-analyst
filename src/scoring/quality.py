"""
Block A — Business Quality Score (0–10).

Вопрос: насколько хорош бизнес сам по себе?

Метрики (из plan.md):
  revenue_growth, eps_growth, gross_margin, operating_margin,
  net_margin, roe, fcf_margin (FCF / Revenue)

Каждая метрика берётся как среднее за последние N лет,
оценивается через Threshold из Benchmark, результаты усредняются.
"""
from __future__ import annotations

import math

from src.data.normalizer import NormalisedData
from src.models.benchmarks import Benchmark
from src.scoring.base import BlockScore, avg_scores


def _recent_mean(values: list[float], n: int = 3) -> float:
    tail = [v for v in values[-n:] if math.isfinite(v)]
    return sum(tail) / len(tail) if tail else float("nan")


def score_quality(nd: NormalisedData, bm: Benchmark) -> BlockScore:
    """
    Compute Business Quality score for *nd* using thresholds from *bm*.

    Returns BlockScore with score in [0, 10].
    """
    breakdown: dict[str, float] = {}
    notes: list[str] = []

    # --- Revenue growth (3-year avg YoY %) ---------------------------------
    rev_g = _recent_mean(nd.revenue_growth_annual)
    s = bm.score_metric("revenue_growth", rev_g)
    if math.isfinite(s):
        breakdown["revenue_growth"] = s
        if rev_g > 20:
            notes.append(f"strong revenue growth {rev_g:.0f}%")
        elif rev_g < 0:
            notes.append(f"declining revenue {rev_g:.0f}%")

    # --- EPS growth (3-year avg) -------------------------------------------
    eps_g = _recent_mean(nd.eps_growth_annual)
    s = bm.score_metric("eps_growth", eps_g)
    if math.isfinite(s):
        breakdown["eps_growth"] = s

    # --- Gross margin (recent avg %) ----------------------------------------
    gm = _recent_mean(nd.gross_margin_annual)
    s = bm.score_metric("gross_margin", gm)
    if math.isfinite(s):
        breakdown["gross_margin"] = s
        if gm > 60:
            notes.append(f"high gross margin {gm:.0f}%")

    # --- Operating margin --------------------------------------------------
    om = _recent_mean(nd.operating_margin_annual)
    s = bm.score_metric("operating_margin", om)
    if math.isfinite(s):
        breakdown["operating_margin"] = s

    # --- Net margin --------------------------------------------------------
    nm = _recent_mean(nd.net_margin_annual)
    s = bm.score_metric("net_margin", nm)
    if math.isfinite(s):
        breakdown["net_margin"] = s

    # --- ROE (net_income / equity %) ---------------------------------------
    roe = _recent_mean(nd.roe_annual)
    s = bm.score_metric("roe", roe)
    if math.isfinite(s):
        breakdown["roe"] = s
        if roe > 30:
            notes.append(f"strong ROE {roe:.0f}%")

    # --- FCF margin (FCF / Revenue) ----------------------------------------
    # Compute per-year FCF margin, then average
    fcf_margins: list[float] = []
    for fcf, rev in zip(nd.fcf_annual, nd.revenue_annual):
        if math.isfinite(fcf) and math.isfinite(rev) and rev > 0:
            fcf_margins.append(fcf / rev * 100.0)
        else:
            fcf_margins.append(float("nan"))
    fcf_m = _recent_mean(fcf_margins)
    s = bm.score_metric("fcf_margin", fcf_m)
    if math.isfinite(s):
        breakdown["fcf_margin"] = s
        if fcf_m > 20:
            notes.append(f"strong FCF margin {fcf_m:.0f}%")
        elif fcf_m < 0:
            notes.append("negative FCF")

    final = avg_scores(breakdown)
    if not breakdown:
        notes.append("insufficient data for quality scoring")

    return BlockScore(score=final, breakdown=breakdown, notes=notes)
