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
from typing import Optional

from src.data.normalizer import NormalisedData
from src.models.benchmarks import Benchmark
from src.scoring.base import BlockScore, avg_scores


def _recent_mean(values: list[float], n: int = 3) -> float:
    tail = [v for v in values[-n:] if math.isfinite(v)]
    return sum(tail) / len(tail) if tail else float("nan")


def _ttm_or_recent(annual: list[float], ttm: "Optional[float]") -> float:
    """Return TTM value when available (most current); fall back to 3-year annual mean.

    TTM data from 10-Q quarterly filings is 3–12 months fresher than the most
    recent annual 10-K figure, so it takes priority for point-in-time scoring.
    """
    if ttm is not None and math.isfinite(ttm):
        return ttm
    return _recent_mean(annual)


def score_quality(nd: NormalisedData, bm: Benchmark) -> BlockScore:
    """
    Compute Business Quality score for *nd* using thresholds from *bm*.

    Returns BlockScore with score in [0, 10].
    Coverage penalty applies when fundamental data is sparse: a company with
    2 available metrics cannot score as well as one with full coverage.
    """
    breakdown: dict[str, float] = {}
    notes: list[str] = []
    attempted = 0

    # --- Revenue growth (3-year avg YoY %) ---------------------------------
    attempted += 1
    rev_g = _recent_mean(nd.revenue_growth_annual)
    s = bm.score_metric("revenue_growth", rev_g)
    if math.isfinite(s):
        breakdown["revenue_growth"] = s
        if rev_g > 20:
            notes.append(f"strong revenue growth {rev_g:.0f}%")
        elif rev_g < 0:
            notes.append(f"declining revenue {rev_g:.0f}%")

    # --- EPS growth (3-year avg) -------------------------------------------
    attempted += 1
    eps_g = _recent_mean(nd.eps_growth_annual)
    s = bm.score_metric("eps_growth", eps_g)
    if math.isfinite(s):
        breakdown["eps_growth"] = s

    # --- Gross margin (TTM preferred, else recent annual avg %) ---------------
    attempted += 1
    gm = _ttm_or_recent(nd.gross_margin_annual, nd.ttm_gross_margin)
    s = bm.score_metric("gross_margin", gm)
    if math.isfinite(s):
        breakdown["gross_margin"] = s
        if gm > 60:
            notes.append(f"high gross margin {gm:.0f}%")

    # --- Operating margin (TTM preferred) ----------------------------------
    attempted += 1
    om = _ttm_or_recent(nd.operating_margin_annual, nd.ttm_operating_margin)
    s = bm.score_metric("operating_margin", om)
    if math.isfinite(s):
        breakdown["operating_margin"] = s

    # --- Net margin (TTM preferred) ----------------------------------------
    attempted += 1
    nm = _ttm_or_recent(nd.net_margin_annual, nd.ttm_net_margin)
    s = bm.score_metric("net_margin", nm)
    if math.isfinite(s):
        breakdown["net_margin"] = s

    # --- ROE (net_income / equity %) ---------------------------------------
    attempted += 1
    roe = _recent_mean(nd.roe_annual)
    s = bm.score_metric("roe", roe)
    if math.isfinite(s):
        breakdown["roe"] = s
        if roe > 30:
            notes.append(f"strong ROE {roe:.0f}%")

    # --- FCF margin (TTM preferred, else per-year annual avg) ---------------
    # Compute per-year FCF margin, then average (annual fallback)
    attempted += 1
    if nd.ttm_fcf_margin is not None and math.isfinite(nd.ttm_fcf_margin):
        fcf_m = nd.ttm_fcf_margin
    else:
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

    coverage = len(breakdown) / attempted if attempted > 0 else 1.0
    final = avg_scores(breakdown, expected_count=attempted)
    if not breakdown:
        notes.append("insufficient data for quality scoring")

    return BlockScore(score=final, breakdown=breakdown, notes=notes, coverage=coverage)
