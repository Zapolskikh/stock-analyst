"""
Engine — Шаги 6–8 алгоритма.

Публичный интерфейс
-------------------
    from src.engine.engine import analyse, AnalysisResult

    result = analyse("NVDA")       # полный пайплайн: данные → оценка → итог
    result.overall_score           # float  0-100
    result.horizon.short           # float  0-100
    result.rating                  # "Strong Candidate" | "Good Candidate" | ...
    result.decision                # "Buy" | "Watch" | "Hold" | "Avoid"
    result.stop_factors            # list[StopFactor]
    result.block_scores            # dict[str, BlockScore]

Архитектура
-----------
  1. Загрузка данных через yfinance / SEC EDGAR (с кешем)
  2. Нормализация → NormalisedData
  3. Классификация типа акции
  4. Выбор бенчмарка
  5. Расчёт 5 блоков оценки (quality / valuation / technical / risk / style_fit)
  6. Горизонтальные оценки (short / medium / long) с разными весами
  7. Проверка стоп-факторов
  8. Итоговый рейтинг + решение

Примечание: для unit-тестирования без внешних запросов используйте
``analyse_nd(nd)`` напрямую.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.data.normalizer import NormalisedData
from src.classifier import classify, CompanyType
from src.models.benchmarks import get_benchmark
from src.models.config_version import current_version
from src.scoring.base import BlockScore
from src.scoring.quality import score_quality
from src.scoring.valuation import score_valuation
from src.scoring.technical import score_technical
from src.scoring.risk import score_risk
from src.scoring.style_fit import score_style_fit
from src.scoring.fair_value import compute_fair_value, FairValueResult


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HorizonScores:
    """Scores 0–100 per investment horizon."""
    short: float   # weeks to ~3 months
    medium: float  # 3–12 months
    long: float    # 1–5 years


@dataclass
class HorizonDecisions:
    """Investment decision per horizon: "Buy" | "Watch" | "Hold" | "Avoid"."""
    short:  str
    medium: str
    long:   str


@dataclass
class StopFactor:
    """A red flag that may override or downgrade the recommendation."""
    name: str
    description: str
    severity: str  # "warning" | "critical"


@dataclass
class AnalysisResult:
    ticker: str
    company_type: CompanyType
    classification_confidence: float
    block_scores: dict[str, BlockScore]
    horizon: HorizonScores
    overall_score: float       # 0–100
    stop_factors: list[StopFactor] = field(default_factory=list)
    rating: str = ""
    decision: str = ""         # medium-horizon decision (backward-compat alias)
    data_confidence: str = "good"   # "good" | "partial" | "poor"
    horizon_decisions: HorizonDecisions = field(
        default_factory=lambda: HorizonDecisions("Hold", "Hold", "Hold")
    )
    config_version: str = ""   # version of scoring thresholds/weights used
    fair_value: Optional[FairValueResult] = None   # intrinsic value estimate


# ---------------------------------------------------------------------------
# Horizon weight tables  (block → weight per horizon)
# ---------------------------------------------------------------------------

_HORIZON_WEIGHTS: dict[str, dict[str, float]] = {
    "short": {
        "quality":   0.10,
        "valuation": 0.20,
        "technical": 0.40,
        "risk":      0.25,
        "style_fit": 0.05,
    },
    "medium": {
        "quality":   0.25,
        "valuation": 0.30,
        "technical": 0.20,
        "risk":      0.15,
        "style_fit": 0.10,
    },
    "long": {
        "quality":   0.35,
        "valuation": 0.15,
        "technical": 0.10,
        "risk":      0.25,
        "style_fit": 0.15,
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _weighted_avg(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted average of scores; missing keys treated as 0."""
    total_w = sum(weights.values())
    if total_w == 0:
        return 0.0
    total = sum(scores.get(k, 0.0) * w for k, w in weights.items())
    return total / total_w


def _rating(score: float) -> str:
    if score >= 85:
        return "Strong Candidate"
    if score >= 70:
        return "Good Candidate"
    if score >= 55:
        return "Neutral / Watchlist"
    if score >= 40:
        return "Weak"
    return "Avoid"


def _decision(score: float, stop_factors: list[StopFactor]) -> str:
    has_critical = any(sf.severity == "critical" for sf in stop_factors)
    if has_critical:
        return "Avoid"
    if score >= 70:
        return "Buy"
    if score >= 55:
        return "Watch"
    if score >= 40:
        return "Hold"
    return "Avoid"


# D/E стоп не применяется к типам с нормальным структурным левериджем
_DE_STOP_EXEMPT = frozenset({CompanyType.FINANCIAL})

# Тип-специфичный порог D/E для критичного стопа
_DE_CRITICAL_THRESHOLD: dict[CompanyType, float] = {
    CompanyType.TURNAROUND: 8.0,    # Turnaround терпит высокий долг
    CompanyType.CYCLICAL:   6.0,    # Cyclical: капиталоёмкий бизнес
}
_DE_CRITICAL_DEFAULT = 4.0


def _check_stop_factors(
    nd: NormalisedData,
    blocks: dict[str, BlockScore],
    company_type: CompanyType = CompanyType.OTHER,
) -> list[StopFactor]:
    """Detect structural red flags that may override a high score.

    *company_type* is used to skip D/E checks for Financial companies
    and to apply type-specific D/E thresholds for Turnaround/Cyclical.
    """
    factors: list[StopFactor] = []

    recent_fcf = [x for x in nd.fcf_annual[-3:] if not math.isnan(x)]
    recent_de = [x for x in nd.debt_to_equity_annual[-2:] if not math.isnan(x)]

    # 1. Persistently negative FCF
    if len(recent_fcf) >= 2 and all(f < 0 for f in recent_fcf):
        # For Financial companies high D/E is normal — don't use D/E to escalate
        if company_type in _DE_STOP_EXEMPT:
            severity = "warning"
        else:
            severity = "critical" if (recent_de and recent_de[-1] > 2.0) else "warning"
        factors.append(StopFactor(
            name="Negative FCF",
            description=f"Persistently negative free cash flow over last {len(recent_fcf)} years",
            severity=severity,
        ))

    # 2. Extreme overvaluation
    if nd.pe_trailing is not None and nd.pe_trailing > 100:
        factors.append(StopFactor(
            name="Extreme Valuation",
            description=f"Trailing P/E = {nd.pe_trailing:.0f} — extremely overvalued",
            severity="warning",
        ))

    # 3. Very high leverage — skipped entirely for Financial sector
    if company_type not in _DE_STOP_EXEMPT:
        de_threshold = _DE_CRITICAL_THRESHOLD.get(company_type, _DE_CRITICAL_DEFAULT)
        if recent_de and recent_de[-1] > de_threshold:
            factors.append(StopFactor(
                name="High Debt",
                description=(
                    f"Debt-to-Equity = {recent_de[-1]:.1f} — "
                    f"exceeds threshold {de_threshold:.0f}x for {company_type.value}"
                ),
                severity="critical",
            ))

    # 4. Sharply deteriorating margins into deep negative
    margins = [x for x in nd.net_margin_annual[-3:] if not math.isnan(x)]
    if len(margins) == 3 and margins[2] < margins[1] < margins[0] and margins[2] < -10:
        factors.append(StopFactor(
            name="Deteriorating Margins",
            description=f"Net margin declining sharply to {margins[2]:.1f}%",
            severity="warning",
        ))

    # 5. Technical breakdown
    tech = blocks.get("technical")
    if tech is not None and tech.score < 3.0:
        factors.append(StopFactor(
            name="Technical Breakdown",
            description="Price in severe downtrend (score < 3/10)",
            severity="warning",
        ))

    # 6. Low liquidity — measured in dollar ADV (avg_volume × current_price)
    # Share-count thresholds are misleading: 100k shares of a $2 stock ($200k/day)
    # is far less liquid than 100k shares of a $500 stock ($50M/day).
    vol = nd.avg_volume
    price = nd.current_price
    dollar_adv: float | None = None
    if vol is not None and math.isfinite(vol) and price is not None and math.isfinite(price) and price > 0:
        dollar_adv = vol * price
    if dollar_adv is not None:
        if dollar_adv < 5_000_000:          # < $5M/day — very illiquid
            factors.append(StopFactor(
                name="Low Liquidity",
                description=f"Dollar ADV ${dollar_adv/1e6:.1f}M/day — very illiquid, wide spreads likely",
                severity="warning",
            ))
        elif dollar_adv < 20_000_000:       # $5M–$20M/day — limited tradability
            factors.append(StopFactor(
                name="Limited Liquidity",
                description=f"Dollar ADV ${dollar_adv/1e6:.1f}M/day — limited tradability",
                severity="warning",
            ))
    elif vol is not None and math.isfinite(vol):
        # Fallback when price unavailable: use share-count thresholds
        if vol < 100_000:
            factors.append(StopFactor(
                name="Low Liquidity",
                description=f"Avg daily volume {vol:,.0f} shares — very illiquid (price unavailable for $ADV)",
                severity="warning",
            ))
        elif vol < 500_000:
            factors.append(StopFactor(
                name="Limited Liquidity",
                description=f"Avg daily volume {vol:,.0f} shares — limited tradability (price unavailable for $ADV)",
                severity="warning",
            ))

    return factors


# ---------------------------------------------------------------------------
# Core analysis function (accepts pre-built NormalisedData)
# ---------------------------------------------------------------------------

def analyse_nd(nd: NormalisedData) -> AnalysisResult:
    """Run full scoring pipeline on an already-normalised dataset.

    This is the testable, pure-logic entry point.
    ``analyse(ticker)`` delegates here after fetching real data.
    """
    cr = classify(nd)
    bm = get_benchmark(cr.company_type)

    blocks: dict[str, BlockScore] = {
        "quality":   score_quality(nd, bm),
        "valuation": score_valuation(nd, bm),
        "technical": score_technical(nd),
        "risk":      score_risk(nd, bm, cr.company_type),
        "style_fit": score_style_fit(nd, bm),
    }

    raw: dict[str, float] = {k: v.score for k, v in blocks.items()}

    # Overall: use benchmark's own block weights (scaled 0–100)
    bw = bm.weights
    overall = _weighted_avg(raw, {
        "quality":   bw.quality,
        "valuation": bw.valuation,
        "technical": bw.technical,
        "risk":      bw.risk,
        "style_fit": bw.style_fit,
    }) * 10.0

    # Horizon scores (0–100)
    short  = _weighted_avg(raw, _HORIZON_WEIGHTS["short"])  * 10.0
    medium = _weighted_avg(raw, _HORIZON_WEIGHTS["medium"]) * 10.0
    long_  = _weighted_avg(raw, _HORIZON_WEIGHTS["long"])   * 10.0

    stop_factors = _check_stop_factors(nd, blocks, cr.company_type)
    rating   = _rating(overall)

    horizon_decisions = HorizonDecisions(
        short=_decision(short,  stop_factors),
        medium=_decision(medium, stop_factors),
        long=_decision(long_,   stop_factors),
    )
    decision = horizon_decisions.medium   # backward-compat alias

    fair_value = compute_fair_value(nd, cr.company_type)

    return AnalysisResult(
        ticker=nd.ticker,
        company_type=cr.company_type,
        classification_confidence=cr.confidence,
        block_scores=blocks,
        horizon=HorizonScores(short=short, medium=medium, long=long_),
        overall_score=overall,
        stop_factors=stop_factors,
        rating=rating,
        decision=decision,
        data_confidence=nd.data_quality,
        horizon_decisions=horizon_decisions,
        config_version=current_version(),
        fair_value=fair_value,
    )


# ---------------------------------------------------------------------------
# Full pipeline (fetches live data)
# ---------------------------------------------------------------------------

def analyse(ticker: str) -> AnalysisResult:
    """Fetch data, normalise, and score a ticker end-to-end.

    Requires network access (yfinance + SEC EDGAR).
    For offline unit tests use ``analyse_nd()`` instead.
    """
    import pandas as pd
    from src.data.price import fetch_ohlcv, fetch_info
    from src.data.sec_edgar import fetch_fundamentals
    from src.data.normalizer import normalise

    ticker = ticker.upper().strip()

    try:
        price_df = fetch_ohlcv(ticker, period="2y")
    except Exception:
        price_df = pd.DataFrame()

    try:
        spy_df = fetch_ohlcv("SPY", period="2y")
        spy_prices = [float(v) for v in spy_df["Close"].dropna().tail(252)]
    except Exception:
        spy_prices = []

    try:
        info = fetch_info(ticker)
    except Exception:
        info = {}

    try:
        fundamentals = fetch_fundamentals(ticker)
    except Exception:
        fundamentals = {}

    nd = normalise(fundamentals, price_df, info, ticker, spy_prices=spy_prices)
    return analyse_nd(nd)
