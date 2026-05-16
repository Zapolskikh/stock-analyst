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

from src.classifier import CompanyType
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


def score_valuation(
    nd: NormalisedData,
    bm: Benchmark,
    company_type: CompanyType = CompanyType.OTHER,
) -> BlockScore:
    """
    Compute Valuation score for *nd* using thresholds from *bm*.
    All scales (P/S, FCF yield, PEG) are benchmark-specific.

    Coverage penalty: pe_trailing, pe_forward, ps_ratio, fcf_yield are always
    expected for a listed company. PEG and EV/EBITDA are supplementary — counted
    only when their prerequisites (pe_forward, ebitda) are available.

    For Financial companies (banks): FCF yield and P/S are not applicable —
    banks do not report revenue / FCF in a comparable way. P/TBV (price-to-book)
    is used as the primary valuation anchor instead.
    """
    is_financial = (company_type == CompanyType.FINANCIAL)
    is_cyclical  = (company_type == CompanyType.CYCLICAL)
    breakdown: dict[str, float] = {}
    notes: list[str] = []
    attempted = 0

    # --- Trailing P/E (always expected for a profitable company) -----------
    # Accounting distortion guard: when pe_trailing > 3× pe_forward AND
    # pe_forward < 25, the trailing figure is almost certainly distorted by
    # one-off GAAP charges (acquisition amortisation, impairments, etc.).
    # In such cases we skip pe_trailing to avoid penalising the valuation score.
    # The stop factor "Accounting Distortion (P/E)" is added by the engine.
    #
    # For CYCLICAL companies pe_trailing is replaced by normalized_pe (median
    # 7yr EPS basis) which smooths commodity and economic cycle swings.
    # Example: XOM pe_trailing = 14x (depressed 2024 earnings) but normalized
    # PE = 10x (mid-cycle) — the normalized figure is the meaningful anchor.
    attempted += 1
    pe_t = nd.pe_trailing
    pe_f_for_guard = nd.pe_forward
    _pe_t_distorted = (
        pe_t is not None and math.isfinite(pe_t) and pe_t > 0
        and pe_f_for_guard is not None and math.isfinite(pe_f_for_guard) and pe_f_for_guard > 0
        and pe_t > 3.0 * pe_f_for_guard and pe_f_for_guard < 25
    )
    if is_cyclical and nd.normalized_pe is not None and math.isfinite(nd.normalized_pe) and nd.normalized_pe > 0:
        # Use normalized (mid-cycle) PE for cyclicals
        s = bm.score_metric("normalized_pe", nd.normalized_pe)
        if math.isfinite(s):
            breakdown["normalized_pe"] = s
            notes.append(
                f"Normalized P/E {nd.normalized_pe:.1f}x (mid-cycle 7yr median EPS) "
                f"— trailing P/E {pe_t:.0f}x excluded for cyclical"
                if pe_t is not None else f"Normalized P/E {nd.normalized_pe:.1f}x (mid-cycle)"
            )
    elif not _pe_t_distorted and pe_t is not None and math.isfinite(pe_t) and pe_t > 0:
        s = bm.score_metric("pe_trailing", pe_t)
        if math.isfinite(s):
            breakdown["pe_trailing"] = s
            if pe_t > 60:
                notes.append(f"P/E {pe_t:.0f} — very expensive")
            elif pe_t < 12:
                notes.append(f"P/E {pe_t:.0f} — cheap")
    elif _pe_t_distorted:
        notes.append(
            f"Trailing P/E {pe_t:.0f} excluded — likely accounting distortion "
            f"(forward P/E {pe_f_for_guard:.1f} is the reliable anchor)"
        )

    # --- Forward P/E (always expected) ------------------------------------
    attempted += 1
    pe_f = nd.pe_forward
    if pe_f is not None and math.isfinite(pe_f) and pe_f > 0:
        s = bm.score_metric("pe_forward", pe_f)
        if math.isfinite(s):
            breakdown["pe_forward"] = s

    # --- P/S ratio (market_cap / TTM revenue preferred, else last annual) ---
    # NOT applicable for Financial sector: bank "revenue" (net interest income)
    # is not comparable to corporate revenue. P/TBV is used instead (see below).
    if not is_financial:
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

    # --- P/TBV — Price to Tangible Book Value (primary metric for banks) ---
    # For Financial companies, P/TBV = market_cap / equity is the standard
    # institutional valuation anchor. Below 1.0 = trading at book discount.
    # For non-financial companies P/TBV has limited meaning and is skipped.
    if is_financial:
        attempted += 1
        last_eq = next((v for v in reversed(nd.equity_annual) if math.isfinite(v) and v > 0), None)
        if nd.market_cap and math.isfinite(nd.market_cap) and last_eq:
            ptbv = nd.market_cap / last_eq
            s = bm.score_metric("ptbv", ptbv)
            if math.isfinite(s):
                breakdown["ptbv"] = s
                if ptbv < 1.0:
                    notes.append(f"P/TBV {ptbv:.2f} — below book value (cheap for a bank)")
                elif ptbv > 3.0:
                    notes.append(f"P/TBV {ptbv:.2f} — premium to book (expensive)")
                else:
                    notes.append(f"P/TBV {ptbv:.2f}")

    # --- FCF yield (FCF / market_cap × 100) — always expected (except banks) --
    # TTM FCF preferred (from 10-Q), else last annual FCF
    # NOT applicable for Financial sector: banks' OCF/FCF is distorted by loan
    # origination cash flows and is not a meaningful valuation input.
    if not is_financial:
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
            # Cap EPS growth used in PEG denominator at 50%.
            # Without this cap, hypergrowth stocks (NVDA: EPS growth 400%+) produce
            # PEG near 0, which reads as "extremely cheap" — but it's a mathematical
            # artefact of the hyper-base, not a genuine valuation signal.
            # Cap of 50% still rewards high-growth companies while preventing
            # near-zero PEG from distorting the valuation score.
            eps_g_capped = min(eps_g, 50.0)
            peg = pe_f / eps_g_capped
            s = bm.score_metric("peg_ratio", peg)
            if math.isfinite(s):
                growth_note = f"{eps_g:.0f}% (capped 50%)" if eps_g > 50.0 else f"{eps_g:.0f}%"
                # Reliability check: explosive EPS growth (>80%) is often
                # cyclical or base-effect driven — PEG signal is unreliable.
                # In such cases: show as informational note only, exclude from score.
                if eps_g > 80.0:
                    notes.append(
                        f"PEG {peg:.2f} (informational, not scored): EPS growth {eps_g:.0f}% "
                        "is base-effect/cyclical — PEG excluded from valuation score"
                    )
                    # Do NOT add to breakdown — score would be misleading
                else:
                    breakdown["peg_ratio"] = s
                    if peg < 1.0:
                        notes.append(f"PEG {peg:.2f} — attractive (growth underpriced, g={growth_note})")
                    elif peg > 3.0:
                        notes.append(f"PEG {peg:.2f} — expensive relative to growth (g={growth_note})")

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
