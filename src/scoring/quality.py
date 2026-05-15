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

from src.classifier import CompanyType
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


def score_quality(
    nd: NormalisedData,
    bm: Benchmark,
    company_type: CompanyType = CompanyType.OTHER,
) -> BlockScore:
    """
    Compute Business Quality score for *nd* using thresholds from *bm*.

    Returns BlockScore with score in [0, 10].
    Coverage penalty applies when fundamental data is sparse: a company with
    2 available metrics cannot score as well as one with full coverage.

    For Financial companies (banks, insurance): gross margin, FCF margin,
    CFO/NI ratio and accruals are NOT applicable — these metrics are
    structurally distorted by banking accounting and are skipped entirely.
    """
    is_financial = (company_type == CompanyType.FINANCIAL)
    breakdown: dict[str, float] = {}
    notes: list[str] = []
    attempted = 0
    if is_financial:
        notes.append("Financial sector: gross margin, FCF margin, CFO/NI and accruals not applicable")

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
    if not is_financial:
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
    # Skip when equity is thin relative to total assets (< 5%) or negative:
    # post-M&A goodwill destroys book equity and makes ROE an artefact of
    # the balance sheet structure, not a signal of capital efficiency.
    # Exception: Financial sector banks structurally have thin equity vs assets
    # (10:1 leverage is normal) — ROE IS the primary profitability metric for banks.
    attempted += 1
    last_eq = next((v for v in reversed(nd.equity_annual) if math.isfinite(v)), None)
    last_ta = next((v for v in reversed(nd.total_assets_annual) if math.isfinite(v)), None)
    roe_reliable = (
        last_eq is not None
        and last_ta is not None
        and last_ta > 0
        and last_eq > 0
        and (is_financial or last_eq / last_ta >= 0.05)  # banks: skip equity ratio check
    )
    if roe_reliable:
        roe = _recent_mean(nd.roe_annual)
        s = bm.score_metric("roe", roe)
        if math.isfinite(s):
            # Cap ROE score at 8.0: ROE > ~100% is almost always a balance-sheet
            # artifact (aggressive buybacks reducing equity, post-M&A goodwill, etc.)
            # and does not reflect true capital efficiency. A real 10/10 ROE would
            # require an independently strong equity base, not a shrunken denominator.
            s = min(s, 8.0)
            breakdown["roe"] = s
            if roe > 30:
                notes.append(f"strong ROE {roe:.0f}%")
    else:
        notes.append("ROE excluded — equity < 5% of assets (post-M&A balance sheet)")

    # --- FCF margin (TTM preferred, else per-year annual avg) ---------------
    # Compute per-year FCF margin, then average (annual fallback)
    # NOT applicable for Financial sector: banks' OCF is structurally negative
    # (loan originations are classified as operating outflows in banking accounting).
    if not is_financial:
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

    # --- Earnings quality: CFO/NI ratio ------------------------------------
    # NOT applicable for Financial sector: banks' OCF is inherently negative
    # (loan issuance = operating cash outflow in GAAP banking accounting).
    ni_val  = nd.ttm_net_income if (nd.ttm_net_income is not None and math.isfinite(nd.ttm_net_income)) \
              else next((v for v in reversed(nd.net_income_annual) if math.isfinite(v)), None)
    ocf_val = nd.ttm_operating_cf if (nd.ttm_operating_cf is not None and math.isfinite(nd.ttm_operating_cf)) \
              else next((v for v in reversed(nd.operating_cf_annual) if math.isfinite(v)), None)
    if not is_financial and ni_val is not None and ocf_val is not None and ni_val != 0:
        attempted += 1
        cfo_ni = ocf_val / ni_val
        if   cfo_ni >= 1.5:  s = 10.0
        elif cfo_ni >= 1.2:  s = 8.5
        elif cfo_ni >= 1.0:  s = 7.0
        elif cfo_ni >= 0.7:  s = 5.5
        elif cfo_ni >= 0.5:  s = 3.5
        elif cfo_ni >= 0.0:  s = 1.5
        else:                s = 0.0   # negative OCF
        breakdown["cfo_ni_ratio"] = s
        if cfo_ni >= 1.2:
            notes.append(f"strong cash conversion CFO/NI={cfo_ni:.2f}")
        elif cfo_ni < 0.7:
            notes.append(f"weak cash conversion CFO/NI={cfo_ni:.2f}")

    # --- Earnings quality: Accruals ratio ----------------------------------
    # NOT applicable for Financial sector: total_assets for banks include
    # the loan book (not comparable to industrial company asset base).
    last_ta = next((v for v in reversed(nd.total_assets_annual) if math.isfinite(v)), None)
    if not is_financial and ni_val is not None and ocf_val is not None and last_ta and last_ta > 0:
        attempted += 1
        accruals_pct = (ni_val - ocf_val) / last_ta * 100
        if   accruals_pct <= -3.0: s = 10.0
        elif accruals_pct <=  0.0: s = 8.5
        elif accruals_pct <=  2.0: s = 6.5
        elif accruals_pct <=  4.0: s = 4.0
        elif accruals_pct <=  6.0: s = 2.0
        else:                      s = 0.0
        breakdown["accruals_ratio"] = s
        if accruals_pct > 4.0:
            notes.append(f"elevated accruals {accruals_pct:.1f}% of assets — earnings quality risk")
        elif accruals_pct < -2.0:
            notes.append(f"low accruals {accruals_pct:.1f}% — high earnings quality")

    coverage = len(breakdown) / attempted if attempted > 0 else 1.0
    final = avg_scores(breakdown, expected_count=attempted)
    if not breakdown:
        notes.append("insufficient data for quality scoring")

    return BlockScore(score=final, breakdown=breakdown, notes=notes, coverage=coverage)
