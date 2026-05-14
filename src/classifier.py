"""
Company type classifier — Шаг 3 алгоритма.

Принимает NormalisedData и возвращает CompanyType + уверенность + сигналы.

Типы (из plan.md):
  HYPERGROWTH_TECH   — быстрорастущие технологические компании
  MATURE_TECH        — зрелые технологические / quality compounder
  PHARMA             — фарма / биотех / медицина
  DIVIDEND_DEFENSIVE — дивидендные / защитные акции
  CYCLICAL           — циклические (energy, materials, industrials, auto)
  FINANCIAL          — банки, страховщики, fintech
  TURNAROUND         — компании в процессе восстановления
  OTHER              — не вписывается ни в одну категорию

Алгоритм — rule-based scoring:
  Для каждого типа считается score 0–100 на основе набора сигналов.
  Побеждает тип с наибольшим score.
  Если score победителя < MIN_CONFIDENCE — возвращается OTHER.

Публичный интерфейс
-------------------
    from src.classifier import classify, CompanyType, ClassificationResult

    result = classify(nd)
    result.company_type      # CompanyType.HYPERGROWTH_TECH
    result.confidence        # 0.0–1.0
    result.scores            # dict[CompanyType, float]
    result.signals           # list[str]  — human-readable объяснение
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.data.normalizer import NormalisedData


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

class CompanyType(str, Enum):
    HYPERGROWTH_TECH   = "Hypergrowth Tech"
    MATURE_TECH        = "Mature Tech"
    PHARMA             = "Pharma / Healthcare"
    DIVIDEND_DEFENSIVE = "Dividend / Defensive"
    CYCLICAL           = "Cyclical"
    FINANCIAL          = "Financial"
    TURNAROUND         = "Turnaround"
    OTHER              = "Other"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    company_type: CompanyType
    confidence: float                          # 0.0–1.0
    scores: dict[CompanyType, float] = field(default_factory=dict)
    signals: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_CONFIDENCE = 0.30   # below this → OTHER

# Sector → CompanyType hints  (yfinance sector strings)
_SECTOR_HINTS: dict[str, CompanyType] = {
    "technology":             CompanyType.MATURE_TECH,
    "communication services": CompanyType.MATURE_TECH,
    "healthcare":             CompanyType.PHARMA,
    "biotechnology":          CompanyType.PHARMA,
    "pharmaceuticals":        CompanyType.PHARMA,
    "financials":             CompanyType.FINANCIAL,
    "financial services":     CompanyType.FINANCIAL,
    "banks":                  CompanyType.FINANCIAL,
    "insurance":              CompanyType.FINANCIAL,
    "energy":                 CompanyType.CYCLICAL,
    "basic materials":        CompanyType.CYCLICAL,
    "materials":              CompanyType.CYCLICAL,
    "industrials":            CompanyType.CYCLICAL,
    "consumer cyclical":      CompanyType.CYCLICAL,
    "utilities":              CompanyType.DIVIDEND_DEFENSIVE,
    "real estate":            CompanyType.DIVIDEND_DEFENSIVE,
    "consumer defensive":     CompanyType.DIVIDEND_DEFENSIVE,
}


# ---------------------------------------------------------------------------
# Helper: safe last-N mean of a list (ignores NaN)
# ---------------------------------------------------------------------------

def _recent_mean(values: list[float], n: int = 3) -> Optional[float]:
    """Mean of the last *n* valid (finite) elements, or None if fewer than 1."""
    tail = [v for v in values[-n:] if math.isfinite(v)]
    return sum(tail) / len(tail) if tail else None


def _last_valid(values: list[float]) -> Optional[float]:
    for v in reversed(values):
        if math.isfinite(v):
            return v
    return None


def _count_positive_growth(growth: list[float]) -> int:
    """Count years with positive YoY growth (skip NaN first element)."""
    return sum(1 for v in growth[1:] if math.isfinite(v) and v > 0)


# ---------------------------------------------------------------------------
# Per-type scoring functions
# ---------------------------------------------------------------------------

def _score_hypergrowth_tech(nd: NormalisedData, signals: list[str]) -> float:
    score = 0.0

    # Revenue growth: avg YoY > 20% → strong signal
    avg_rev_growth = _recent_mean(nd.revenue_growth_annual)
    if avg_rev_growth is not None:
        if avg_rev_growth >= 30:
            score += 35; signals.append(f"revenue growth {avg_rev_growth:.0f}% (hypergrowth)")
        elif avg_rev_growth >= 20:
            score += 25; signals.append(f"revenue growth {avg_rev_growth:.0f}% (strong)")
        elif avg_rev_growth >= 10:
            score += 10

    # Gross margin > 50 % typical for software/chips
    avg_gm = _recent_mean(nd.gross_margin_annual)
    if avg_gm is not None:
        if avg_gm >= 60:
            score += 20; signals.append(f"gross margin {avg_gm:.0f}% (very high)")
        elif avg_gm >= 45:
            score += 12

    # Low / no dividend
    div = nd.dividend_yield
    if div is None or div < 0.01:
        score += 10; signals.append("no / minimal dividend (growth profile)")

    # P/E elevated
    pe = nd.pe_trailing or nd.pe_forward
    if pe is not None and pe > 35:
        score += 10; signals.append(f"P/E {pe:.0f} (elevated, growth premium)")

    # Sector hint: подтверждает, но не создаёт классификацию.
    # Выдаётся только если уже есть хотя бы один финансовый сигнал (score > 0 до этой точки)
    if score > 0:
        sector = (nd.sector or "").lower()
        if "technology" in sector or "software" in sector or "semiconductor" in sector:
            score += 8; signals.append(f"sector: {nd.sector}")

    return min(score, 100.0)


def _score_mature_tech(nd: NormalisedData, signals: list[str]) -> float:
    score = 0.0

    avg_rev_growth = _recent_mean(nd.revenue_growth_annual)
    if avg_rev_growth is not None:
        if 5 <= avg_rev_growth < 20:
            score += 25; signals.append(f"moderate revenue growth {avg_rev_growth:.0f}%")
        elif avg_rev_growth < 5:
            score += 10

    avg_gm = _recent_mean(nd.gross_margin_annual)
    if avg_gm is not None and avg_gm >= 40:
        score += 20; signals.append(f"gross margin {avg_gm:.0f}% (quality business)")

    # Consistent FCF
    fcf_positive = sum(1 for v in nd.fcf_annual if math.isfinite(v) and v > 0)
    if fcf_positive >= 3:
        score += 20; signals.append("consistent positive FCF")

    # P/E moderate
    pe = nd.pe_trailing or nd.pe_forward
    if pe is not None and 15 <= pe <= 35:
        score += 10

    # Sector hint: подтверждает, но не создаёт классификацию.
    # Не выдаётся:
    #   • если нет ни одного финансового сигнала (score == 0)
    #   • если чистая маржа отрицательная (убыточная компания — это Turnaround, не Mature)
    if score > 0:
        avg_nm = _recent_mean(nd.net_margin_annual)
        is_losing_money = avg_nm is not None and avg_nm < -2.0
        if not is_losing_money:
            sector = (nd.sector or "").lower()
            if "technology" in sector or "communication" in sector:
                score += 8; signals.append(f"sector: {nd.sector}")

    return min(score, 100.0)


def _score_pharma(nd: NormalisedData, signals: list[str]) -> float:
    score = 0.0

    sector = (nd.sector or "").lower()
    industry = (nd.industry or "").lower()
    if any(k in sector + industry for k in ("health", "pharma", "biotech", "medical", "drug")):
        score += 40; signals.append(f"sector/industry: {nd.sector} / {nd.industry}")

    # R&D intensity (R&D / Revenue)
    rd_ratio = None
    rd = _last_valid(nd.rd_expense_annual)
    rev = _last_valid(nd.revenue_annual)
    if rd is not None and rev is not None and rev > 0:
        rd_ratio = rd / rev * 100
    if rd_ratio is not None:
        if rd_ratio >= 15:
            score += 25; signals.append(f"R&D/Revenue {rd_ratio:.0f}% (pharma profile)")
        elif rd_ratio >= 8:
            score += 12

    # High gross margins common in pharma
    avg_gm = _recent_mean(nd.gross_margin_annual)
    if avg_gm is not None and avg_gm >= 55:
        score += 15; signals.append(f"gross margin {avg_gm:.0f}%")

    return min(score, 100.0)


def _score_dividend_defensive(nd: NormalisedData, signals: list[str]) -> float:
    score = 0.0

    # Strong dividend yield
    div = nd.dividend_yield
    if div is not None:
        if div >= 0.03:
            score += 35; signals.append(f"dividend yield {div * 100:.1f}%")
        elif div >= 0.015:
            score += 20; signals.append(f"dividend yield {div * 100:.1f}%")

    # Low revenue growth
    avg_rev_growth = _recent_mean(nd.revenue_growth_annual)
    if avg_rev_growth is not None and avg_rev_growth < 8:
        score += 15; signals.append(f"low revenue growth {avg_rev_growth:.0f}% (defensive)")

    # Sector
    sector = (nd.sector or "").lower()
    if any(k in sector for k in ("utilities", "consumer defensive", "real estate", "staples")):
        score += 25; signals.append(f"sector: {nd.sector}")

    # Stable earnings
    pos_ni = sum(1 for v in nd.net_income_annual if math.isfinite(v) and v > 0)
    if pos_ni == len([v for v in nd.net_income_annual if math.isfinite(v)]) and pos_ni >= 3:
        score += 15; signals.append("consistently positive net income")

    return min(score, 100.0)


def _score_cyclical(nd: NormalisedData, signals: list[str]) -> float:
    score = 0.0

    sector = (nd.sector or "").lower()
    if any(k in sector for k in ("energy", "material", "industrial", "consumer cyclical", "auto")):
        score += 35; signals.append(f"sector: {nd.sector}")

    # Volatile revenue growth
    growth_vals = [v for v in nd.revenue_growth_annual[1:] if math.isfinite(v)]
    if len(growth_vals) >= 2:
        spread = max(growth_vals) - min(growth_vals)
        if spread >= 30:
            score += 25; signals.append(f"revenue growth spread {spread:.0f}pp (cyclical)")
        elif spread >= 15:
            score += 12

    # Low margins
    avg_net_margin = _recent_mean(nd.net_margin_annual)
    if avg_net_margin is not None and avg_net_margin < 10:
        score += 15; signals.append(f"net margin {avg_net_margin:.0f}% (thin, cyclical)")

    return min(score, 100.0)


def _score_financial(nd: NormalisedData, signals: list[str]) -> float:
    score = 0.0

    sector = (nd.sector or "").lower()
    if any(k in sector for k in ("financial", "bank", "insurance", "credit")):
        score += 50; signals.append(f"sector: {nd.sector}")

    # Financials often have high liabilities (leverage by design)
    ltd = _last_valid(nd.long_term_debt_annual)
    eq  = _last_valid(nd.equity_annual)
    if ltd is not None and eq is not None and eq > 0 and ltd / eq > 3:
        score += 20; signals.append(f"high leverage (D/E {ltd / eq:.1f}x) — financial profile")

    return min(score, 100.0)


def _score_turnaround(nd: NormalisedData, signals: list[str]) -> float:
    score = 0.0

    # Negative net income in recent years
    neg_ni = sum(1 for v in nd.net_income_annual[-3:] if math.isfinite(v) and v < 0)
    if neg_ni >= 2:
        score += 35; signals.append(f"{neg_ni} of last 3 years with negative net income")
    elif neg_ni == 1:
        score += 15

    # Recovering revenue after decline
    growth_vals = [v for v in nd.revenue_growth_annual if math.isfinite(v)]
    if len(growth_vals) >= 2:
        had_decline = any(v < -5 for v in growth_vals[:-1])
        recovering  = growth_vals[-1] > 5 if growth_vals else False
        if had_decline and recovering:
            score += 30; signals.append("revenue declined then recovering")

    # High debt
    de = _last_valid(nd.debt_to_equity_annual)
    if de is not None and de > 2.0:
        score += 15; signals.append(f"high debt-to-equity {de:.1f}x")

    # Если компания явно убыточная (независимо от сектора) — дополнительный бонус,
    # чтобы конкурировать с Mature Tech / Cyclical, которые дают бонус за сектор
    avg_nm = _recent_mean(nd.net_margin_annual)
    if avg_nm is not None and avg_nm < -2.0 and neg_ni >= 1:
        score += 10; signals.append(f"negative net margin {avg_nm:.1f}% confirms distress")

    return min(score, 100.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(nd: NormalisedData) -> ClassificationResult:
    """
    Classify *nd* into one of the CompanyType categories.

    Returns a ClassificationResult with the winning type, confidence,
    per-type scores and human-readable signals.
    """
    all_signals: dict[CompanyType, list[str]] = {t: [] for t in CompanyType}

    scorers = {
        CompanyType.HYPERGROWTH_TECH:   _score_hypergrowth_tech,
        CompanyType.MATURE_TECH:        _score_mature_tech,
        CompanyType.PHARMA:             _score_pharma,
        CompanyType.DIVIDEND_DEFENSIVE: _score_dividend_defensive,
        CompanyType.CYCLICAL:           _score_cyclical,
        CompanyType.FINANCIAL:          _score_financial,
        CompanyType.TURNAROUND:         _score_turnaround,
    }

    scores: dict[CompanyType, float] = {}
    for ctype, fn in scorers.items():
        sig: list[str] = []
        scores[ctype] = fn(nd, sig)
        all_signals[ctype] = sig
    scores[CompanyType.OTHER] = 0.0

    # Normalise to 0–1
    max_score = max(scores.values()) or 1.0
    confidences = {t: s / 100.0 for t, s in scores.items()}

    winner = max(scores, key=lambda t: scores[t])
    confidence = confidences[winner]

    if confidence < MIN_CONFIDENCE:
        winner = CompanyType.OTHER
        confidence = 0.0

    return ClassificationResult(
        company_type=winner,
        confidence=round(confidence, 3),
        scores={t: round(s, 1) for t, s in scores.items()},
        signals=all_signals.get(winner, []),
    )
