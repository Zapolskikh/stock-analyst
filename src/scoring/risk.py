"""
Block D — Risk Score (0–10), где 10 = минимальный риск.

Вопрос: какие риски могут испортить инвестиционную идею?

Метрики:
  debt_to_equity      — долговая нагрузка (пропускается для Financial-сектора)
  beta                — рыночный риск
  earnings_stability  — стабильность чистой маржи (robust, не взрывается при нуле)
  fcf_consistency     — доля лет с положительным FCF
  revenue_stability   — стабильность роста выручки
"""
from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Optional

from src.classifier import CompanyType
from src.data.normalizer import NormalisedData
from src.models.benchmarks import Benchmark
from src.scoring.base import BlockScore, avg_scores

# Типы компаний, для которых D/E структурно нерелевантен
_DE_EXEMPT_TYPES = frozenset({
    CompanyType.FINANCIAL,
})


def _last_valid(values: list[float]) -> float:
    for v in reversed(values):
        if math.isfinite(v):
            return v
    return float("nan")


def _last_valid_pair(a: list[float], b: list[float]) -> tuple[float, float]:
    """Return the most recent (a_val, b_val) pair where both are finite, same index."""
    for i in range(min(len(a), len(b)) - 1, -1, -1):
        if math.isfinite(a[i]) and math.isfinite(b[i]):
            return a[i], b[i]
    return float("nan"), float("nan")


def _earnings_instability(values: list[float]) -> float:
    """
    Robust instability measure for a net-margin series.

    Replaces naive CV (std / |mean|) which blows up when mean ≈ 0
    (e.g. Turnaround companies transitioning from loss to profit).

    Formula: std / max(|mean|, median_abs, 1.0)
      — median_abs anchors the scale when mean is near zero
      — 1.0 as a floor prevents division by near-zero
    Returns NaN if fewer than 2 finite values.
    """
    valid = [v for v in values if math.isfinite(v)]
    if len(valid) < 2:
        return float("nan")
    m = mean(valid)
    s = stdev(valid)
    abs_values = sorted(abs(v) for v in valid)
    median_abs = abs_values[len(abs_values) // 2]
    ref = max(abs(m), median_abs, 1.0)
    return s / ref


def _coeff_variation(values: list[float]) -> float:
    """CV для revenue growth — mean обычно далеко от нуля, взрыва нет."""
    valid = [v for v in values if math.isfinite(v)]
    if len(valid) < 2:
        return float("nan")
    m = mean(valid)
    if abs(m) < 0.5:   # growth ≈ 0 — избегаем деления на ~0
        return float("nan")
    return stdev(valid) / abs(m)


def score_risk(
    nd: NormalisedData,
    bm: Benchmark,
    company_type: CompanyType = CompanyType.OTHER,
) -> BlockScore:
    """
    Compute Risk score for *nd* using thresholds from *bm*.
    Higher score = lower risk.

    *company_type* is used to skip D/E for Financial companies where
    structural leverage is normal and D/E > 4 is not a meaningful risk signal.

    Coverage penalty: up to 6 risk metrics expected. Missing data reduces the
    score via sqrt(coverage) to prevent low-data companies from appearing safe.
    """
    breakdown: dict[str, float] = {}
    notes: list[str] = []
    attempted = 0

    # --- Debt / Equity (пропускаем для Financial — леверидж структурный) ---
    if company_type not in _DE_EXEMPT_TYPES:
        attempted += 1
        de = _last_valid(nd.debt_to_equity_annual)
        # Guard: D/E is misleading when equity is negative or near zero.
        _, last_equity = _last_valid_pair(nd.long_term_debt_annual, nd.equity_annual)
        if math.isfinite(last_equity) and last_equity <= 0:
            notes.append("negative book equity — D/E not meaningful; check debt/assets")
            # Attempted was counted -> coverage penalty applies; no breakdown entry
        else:
            s = bm.score_metric("debt_to_equity", de)
            if math.isfinite(s):
                breakdown["debt_to_equity"] = s
                if math.isfinite(de):
                    if de > 3.0:
                        notes.append(f"high D/E {de:.1f}x — elevated leverage risk")
                    elif de < 0.3:
                        notes.append(f"low D/E {de:.2f}x — strong balance sheet")

    # --- Beta (market risk) ------------------------------------------------
    attempted += 1
    beta = nd.beta
    if beta is not None and math.isfinite(beta):
        s = bm.score_metric("beta", beta)
        if math.isfinite(s):
            breakdown["beta"] = s
            if beta > 2.0:
                notes.append(f"beta {beta:.1f} — high volatility")
            elif beta < 0.7:
                notes.append(f"beta {beta:.1f} — low market sensitivity")

    # --- Earnings stability (robust instability, safe при переходе через 0) --
    nm_valid = [v for v in nd.net_margin_annual if math.isfinite(v)]
    if len(nm_valid) >= 2:
        attempted += 1
        instability = _earnings_instability(nm_valid)
        if math.isfinite(instability):
            # instability=0 → perfect (10), =1 → very unstable (2), >2 → (0)
            pts = [(0.0, 10), (0.3, 8), (0.7, 5), (1.0, 2), (2.0, 0)]
            s = _interp(pts, instability)
            if math.isfinite(s):
                breakdown["earnings_stability"] = s
                if instability > 0.8:
                    notes.append(f"earnings instability (score={instability:.2f})")

    # --- FCF consistency (fraction of years with positive FCF) -------------
    fcf_valid = [v for v in nd.fcf_annual if math.isfinite(v)]
    if fcf_valid:
        attempted += 1
        pos_fraction = sum(1 for v in fcf_valid if v > 0) / len(fcf_valid)
        # 100% → 10, 75% → 7, 50% → 4, 0% → 0
        pts = [(0.0, 0), (0.5, 4), (0.75, 7), (1.0, 10)]
        s = _interp(pts, pos_fraction)
        if math.isfinite(s):
            breakdown["fcf_consistency"] = s
            if pos_fraction < 0.5:
                notes.append("FCF negative in majority of years")

    # --- Revenue stability (CV of revenue growth, lower = more stable) ------
    rg_valid = [v for v in nd.revenue_growth_annual[1:] if math.isfinite(v)]
    if len(rg_valid) >= 2:
        attempted += 1
        cv_rg = _coeff_variation(rg_valid)
        if math.isfinite(cv_rg):
            # Same scale as earnings_stability
            pts = [(0.0, 10), (0.5, 7), (1.0, 4), (2.0, 1), (3.0, 0)]
            s = _interp(pts, cv_rg)
            if math.isfinite(s):
                breakdown["revenue_stability"] = s
                if cv_rg > 1.5:
                    notes.append(f"highly volatile revenue growth (CV={cv_rg:.1f})")

    # --- Dilution risk (shares outstanding growth) -------------------------
    # Avg annual dilution > 0% bad; > 5% very bad (убыточные компании размывают)
    dil_valid = [v for v in nd.shares_dilution_annual[1:] if math.isfinite(v)]
    if len(dil_valid) >= 2:
        attempted += 1
        avg_dil = sum(dil_valid) / len(dil_valid)
        # Only penalise dilution (positive growth = more shares = worse for investors)
        # Buybacks (negative growth) scored well
        pts = [(-5.0, 10), (-1.0, 9), (0.0, 8), (2.0, 6), (5.0, 3), (10.0, 0)]
        s = _interp(pts, avg_dil)
        if math.isfinite(s):
            breakdown["dilution_risk"] = s
            if avg_dil > 5.0:
                notes.append(f"significant share dilution ({avg_dil:.1f}%/yr avg)")
            elif avg_dil < -1.0:
                notes.append(f"share buybacks ({avg_dil:.1f}%/yr avg — positive)")

    if not breakdown:
        notes.append("insufficient data for risk scoring")
        return BlockScore(score=5.0, breakdown={}, notes=notes, coverage=0.0)

    coverage = len(breakdown) / attempted if attempted > 0 else 1.0
    final = avg_scores(breakdown, expected_count=attempted)
    return BlockScore(score=final, breakdown=breakdown, notes=notes, coverage=coverage)


def _interp(points: list[tuple[float, float]], value: float) -> float:
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
