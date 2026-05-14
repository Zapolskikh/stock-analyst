"""
Block B — Valuation Score (0–10).

Вопрос: насколько акция дорого или дёшево стоит?

Метрики:
  pe_trailing  — из yfinance
  pe_forward   — из yfinance
  ps_ratio     — market_cap / revenue (latest), шкала из бенчмарка
  fcf_yield    — FCF (latest) / market_cap × 100, шкала из бенчмарка
  peg_ratio    — pe_forward / eps_growth_avg (3y), шкала из бенчмарка
               — актуален для growth-компаний; если eps_growth ≤ 0 — пропускается

Все шкалы тип-специфичны: P/S 8 — дорого для ритейла, нормально для SaaS.
"""
from __future__ import annotations

import math

from src.data.normalizer import NormalisedData
from src.models.benchmarks import Benchmark
from src.scoring.base import BlockScore, avg_scores


def _last_valid(values: list[float]) -> float:
    for v in reversed(values):
        if math.isfinite(v):
            return v
    return float("nan")


def _recent_mean(values: list[float], n: int = 3) -> float:
    tail = [v for v in values[-n:] if math.isfinite(v)]
    return sum(tail) / len(tail) if tail else float("nan")


def score_valuation(nd: NormalisedData, bm: Benchmark) -> BlockScore:
    """
    Compute Valuation score for *nd* using thresholds from *bm*.
    All scales (P/S, FCF yield, PEG) are benchmark-specific.

    Coverage penalty: pe_trailing, pe_forward, ps_ratio, fcf_yield are always
    expected for a listed company. PEG and EV/EBITDA are supplementary — counted
    only when their prerequisites (pe_forward, ebitda) are available.
    """
    breakdown: dict[str, float] = {}
    notes: list[str] = []
    attempted = 0

    # --- Trailing P/E (always expected for a profitable company) -----------
    attempted += 1
    pe_t = nd.pe_trailing
    if pe_t is not None and math.isfinite(pe_t) and pe_t > 0:
        s = bm.score_metric("pe_trailing", pe_t)
        if math.isfinite(s):
            breakdown["pe_trailing"] = s
            if pe_t > 60:
                notes.append(f"P/E {pe_t:.0f} — very expensive")
            elif pe_t < 12:
                notes.append(f"P/E {pe_t:.0f} — cheap")

    # --- Forward P/E (always expected) ------------------------------------
    attempted += 1
    pe_f = nd.pe_forward
    if pe_f is not None and math.isfinite(pe_f) and pe_f > 0:
        s = bm.score_metric("pe_forward", pe_f)
        if math.isfinite(s):
            breakdown["pe_forward"] = s

    # --- P/S ratio (market_cap / TTM revenue preferred, else last annual) ---
    attempted += 1
    # TTM revenue is 3–12 months fresher than the last annual 10-K figure
    last_rev = nd.ttm_revenue if (nd.ttm_revenue is not None and math.isfinite(nd.ttm_revenue)) \
               else _last_valid(nd.revenue_annual)
    if (nd.market_cap and math.isfinite(nd.market_cap)
            and math.isfinite(last_rev) and last_rev > 0):
        ps = nd.market_cap / last_rev
        s = bm.score_metric("ps_ratio", ps)
        if math.isfinite(s):
            breakdown["ps_ratio"] = s
            if ps > 10:
                notes.append(f"P/S {ps:.1f} — elevated")
            elif ps < 1.5:
                notes.append(f"P/S {ps:.1f} — attractive")

    # --- FCF yield (FCF / market_cap × 100) — always expected -------------
    # TTM FCF preferred (from 10-Q), else last annual FCF
    attempted += 1
    last_fcf = nd.ttm_fcf if (nd.ttm_fcf is not None and math.isfinite(nd.ttm_fcf)) \
               else _last_valid(nd.fcf_annual)
    if (nd.market_cap and math.isfinite(nd.market_cap) and nd.market_cap > 0
            and math.isfinite(last_fcf)):
        fcf_yield = last_fcf / nd.market_cap * 100.0
        s = bm.score_metric("fcf_yield", fcf_yield)
        if math.isfinite(s):
            breakdown["fcf_yield"] = s
            if fcf_yield > 5:
                notes.append(f"FCF yield {fcf_yield:.1f}% — attractive")
            elif fcf_yield < 0:
                notes.append("negative FCF yield")

    # --- PEG ratio (pe_forward / eps_growth_avg) ---------------------------
    # Supplementary — only counted when pe_forward and positive growth are present.
    # Semantically not applicable for loss-making or declining companies.
    if pe_f is not None and math.isfinite(pe_f) and pe_f > 0:
        eps_g = _recent_mean(nd.eps_growth_annual)
        if math.isfinite(eps_g) and eps_g > 0:
            attempted += 1
            peg = pe_f / eps_g
            s = bm.score_metric("peg_ratio", peg)
            if math.isfinite(s):
                breakdown["peg_ratio"] = s
                if peg < 1.0:
                    notes.append(f"PEG {peg:.2f} — attractive (growth underpriced)")
                elif peg > 3.0:
                    notes.append(f"PEG {peg:.2f} — expensive relative to growth")

    # --- EV / EBITDA -------------------------------------------------------
    # EV = market_cap + last_long_term_debt − last_cash
    # EBITDA: TTM (operating_income + D&A from 10-Q) preferred; else last annual.
    # Supplementary — counted when EBITDA data is available.
    last_ltd  = _last_valid(nd.long_term_debt_annual)
    last_cash = _last_valid(nd.cash_annual) if nd.cash_annual else float("nan")
    # TTM EBITDA proxy: ttm_operating_income + last annual D&A (D&A is typically stable)
    ttm_oi = nd.ttm_operating_income
    last_da = _last_valid(nd.da_annual) if nd.da_annual else float("nan")
    if ttm_oi is not None and math.isfinite(ttm_oi) and math.isfinite(last_da):
        last_ebitda = ttm_oi + last_da   # TTM EBITDA proxy
    else:
        last_ebitda = _last_valid(nd.ebitda_annual) if nd.ebitda_annual else float("nan")
    if math.isfinite(last_ebitda) and last_ebitda > 0:
        attempted += 1
        if (nd.market_cap and math.isfinite(nd.market_cap)
                and math.isfinite(last_ltd)):
            cash_val = last_cash if math.isfinite(last_cash) else 0.0
            ev = nd.market_cap + last_ltd - cash_val
            if ev > 0:
                ev_ebitda = ev / last_ebitda
                s = bm.score_metric("ev_to_ebitda", ev_ebitda)
                if math.isfinite(s):
                    breakdown["ev_to_ebitda"] = s
                    if ev_ebitda < 8:
                        notes.append(f"EV/EBITDA {ev_ebitda:.1f} — cheap")
                    elif ev_ebitda > 40:
                        notes.append(f"EV/EBITDA {ev_ebitda:.1f} — very expensive")

    coverage = len(breakdown) / attempted if attempted > 0 else 1.0
    final = avg_scores(breakdown, expected_count=attempted)
    if not breakdown:
        notes.append("insufficient valuation data")

    return BlockScore(score=final, breakdown=breakdown, notes=notes, coverage=coverage)


def _interp(points: list[tuple[float, float]], value: float) -> float:
    """Simple linear interpolation (ascending x)."""
    if not math.isfinite(value) or not points:
        return float("nan")
    if value <= points[0][0]:
        return points[0][1]
    if value >= points[-1][0]:
        return points[-1][1]
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x0 <= value <= x1:
            t = (value - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return float("nan")
