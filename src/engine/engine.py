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
class TradeRecommendation:
    """
    Actionable trading recommendation derived from the full analysis.

    action      : "Accumulate" | "Accumulate on Pullback" | "Avoid"
    entry_type  : "market" | "limit" | "avoid"
    limit_price : target entry price for limit orders (None for market)
    limit_wait_days : calendar days to keep the limit order alive
    horizon_label   : "medium (3\u201312 months)" | "long (1\u20135 years)"
    hold_months     : suggested maximum holding period in months
    target_price    : profit-take price
    stop_price      : hard stop-loss price
    rationale       : list of human-readable reasons
    """
    action: str                       # see docstring above
    entry_type: str                   # "market" | "limit" | "avoid"
    horizon_label: str = ""           # "medium (3\u201312 months)" | "long (1\u20135 years)"
    hold_months: Optional[int] = None
    limit_price: Optional[float] = None
    limit_wait_days: Optional[int] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    rationale: list[str] = field(default_factory=list)


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
    trade_rec: Optional[TradeRecommendation] = None  # actionable trade recommendation


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
    if score >= 60:
        return "Borderline"
    if score >= 40:
        return "Weak"
    return "Avoid"


def _decision(score: float, stop_factors: list[StopFactor]) -> str:
    has_critical = any(sf.severity == "critical" for sf in stop_factors)
    if has_critical or score < 60:
        return "Avoid"
    if score >= 70:
        return "Buy"
    return "Buy on Limit"


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
    # Only flag when forward P/E is also elevated or unavailable.
    # Trailing P/E > 100 is often a GAAP artefact: acquisition amortisation,
    # impairment charges, milestone payments — all inflate trailing P/E without
    # indicating true overvaluation (e.g. ABBV post-Allergan, pharma M&A).
    # If forward P/E ≤ 30, analysts already price in the recovery → not a stop.
    if nd.pe_trailing is not None and nd.pe_trailing > 100:
        fwd_pe_ok = (nd.pe_forward is not None
                     and math.isfinite(nd.pe_forward)
                     and nd.pe_forward <= 30)
        if not fwd_pe_ok:
            factors.append(StopFactor(
                name="Extreme Valuation",
                description=(
                    f"Trailing P/E = {nd.pe_trailing:.0f} — extremely overvalued"
                    + (f" (forward P/E = {nd.pe_forward:.1f} also elevated)"
                       if nd.pe_forward is not None else " (no forward P/E available)")
                ),
                severity="warning",
            ))

    # 3. Very high leverage — skipped entirely for Financial sector
    # Also skipped when equity is thin relative to total assets (< 5%):
    # post-M&A goodwill and acquisition amortisation mechanically destroy
    # book equity without indicating real financial distress (ABBV, ORCL).
    # In such cases D/E is unreliable — use Net Debt/EBITDA or interest
    # coverage instead (future enhancement).
    if company_type not in _DE_STOP_EXEMPT:
        # Check equity reliability: equity / total_assets
        last_eq = next((v for v in reversed(nd.equity_annual) if math.isfinite(v)), None)
        last_ta = next((v for v in reversed(nd.total_assets_annual) if math.isfinite(v)), None)
        thin_equity = (
            last_eq is not None
            and last_ta is not None
            and last_ta > 0
            and (last_eq <= 0 or last_eq / last_ta < 0.05)
        )
        de_threshold = _DE_CRITICAL_THRESHOLD.get(company_type, _DE_CRITICAL_DEFAULT)
        if recent_de and recent_de[-1] > de_threshold:
            if thin_equity:
                # Equity is too small / negative to make D/E meaningful
                factors.append(StopFactor(
                    name="Thin Equity",
                    description=(
                        f"Equity = {last_eq/1e9:.1f}B is {last_eq/last_ta*100:.1f}% of assets — "
                        "D/E ratio unreliable (post-M&A goodwill or accumulated losses); "
                        "verify debt service capacity via EBITDA/FCF"
                    ),
                    severity="warning",
                ))
            else:
                factors.append(StopFactor(
                    name="High Debt",
                    description=(
                        f"Debt-to-Equity = {recent_de[-1]:.1f} — "
                        f"exceeds threshold {de_threshold:.0f}x for {company_type.value}"
                    ),
                    severity="critical",
                ))

    # 4a. Accounting distortion: trailing P/E >> forward P/E
    # Common in pharma after acquisitions (ABBV, PFE), companies with impairments,
    # or one-off charges. Trailing P/E becomes misleading — forward P/E is anchored
    # to normalised earnings and is the more reliable signal.
    # Trigger: pe_trailing > 3× pe_forward AND pe_forward < 25.
    _pe_t = nd.pe_trailing
    _pe_f = nd.pe_forward
    if (
        _pe_t is not None and math.isfinite(_pe_t) and _pe_t > 0
        and _pe_f is not None and math.isfinite(_pe_f) and _pe_f > 0
        and _pe_t > 3.0 * _pe_f and _pe_f < 25
    ):
        factors.append(StopFactor(
            name="Accounting Distortion (P/E)",
            description=(
                f"Trailing P/E={_pe_t:.0f} vs forward P/E={_pe_f:.1f} — "
                "large divergence suggests one-off charges, acquisition amortisation "
                "or impairments; trailing P/E excluded from valuation scoring"
            ),
            severity="warning",
        ))

    # 4b. Cashflow anomaly: TTM OCF deviates sharply from 5yr historical average.
    # Detected in normalizer; surfaced here as a stop factor to alert the
    # engine, reduce DCF confidence, and flag to the AI layer.
    if nd.cashflow_anomaly and nd.cashflow_anomaly_detail:
        factors.append(StopFactor(
            name="Cashflow Anomaly",
            description=nd.cashflow_anomaly_detail,
            severity="warning",
        ))

    # 4. Valuation stretched: high P/E, high PEG, and valuation block confirms it.
    # This is distinct from Extreme Valuation (P/E > 100): captures the case where
    # a quality company is priced to perfection — P/E > 30, PEG > 3, and the
    # valuation scoring block independently scores the stock as expensive (< 3/10).
    # All three conditions must hold to avoid false positives on fast-growers.
    val_block = blocks.get("valuation")
    val_score = val_block.score if val_block is not None else 5.0
    pe_t = nd.pe_trailing
    pe_f = nd.pe_forward
    # Compute PEG from pe_forward and recent eps growth
    _eps_g_vals = [v for v in nd.eps_growth_annual[-3:] if math.isfinite(v) and v > 0]
    _peg_check: float | None = None
    if pe_f is not None and math.isfinite(pe_f) and pe_f > 0 and _eps_g_vals:
        _peg_check = pe_f / (sum(_eps_g_vals) / len(_eps_g_vals))
    if (
        pe_t is not None and math.isfinite(pe_t) and pe_t > 30
        and (_peg_check is not None and _peg_check > 3.0)
        and val_score < 3.0
    ):
        factors.append(StopFactor(
            name="Valuation Stretched",
            description=(
                f"P/E={pe_t:.0f}, PEG={_peg_check:.1f} — priced for perfection; "
                "valuation block confirms limited upside at current price"
            ),
            severity="warning",
        ))

    # 5. Overheating / overextension warnings.
    # These do NOT block a buy — they are signals that entry risk is elevated.
    # P/S > 25: any sector at this multiple is priced for extraordinary perfection.
    # MA50 > 20% above price: technically overextended, mean-reversion risk.
    # Beta > 2.0: high-volatility name — drawdowns can be severe.

    # P/S overheating
    _last_rev = nd.ttm_revenue
    if _last_rev is None or not math.isfinite(_last_rev):
        _last_rev = next((v for v in reversed(nd.revenue_annual) if math.isfinite(v) and v > 0), None)
    if nd.market_cap and math.isfinite(nd.market_cap) and _last_rev and _last_rev > 0:
        _ps = nd.market_cap / _last_rev
        if _ps > 25:
            factors.append(StopFactor(
                name="Premium Valuation (P/S)",
                description=(
                    f"P/S ratio {_ps:.1f}x — priced for extraordinary growth; "
                    "any slowdown or margin miss could trigger sharp re-rating"
                ),
                severity="warning",
            ))

    # MA50 overextension
    _price = nd.current_price
    if _price and math.isfinite(_price) and len(nd.close_prices) >= 50:
        _ma50 = sum(nd.close_prices[-50:]) / 50
        if _ma50 > 0 and _price > _ma50 * 1.20:
            _ext = (_price / _ma50 - 1) * 100
            factors.append(StopFactor(
                name="Technically Overextended",
                description=(
                    f"Price {_ext:.0f}% above MA50 — overextended; "
                    "elevated mean-reversion risk in short term"
                ),
                severity="warning",
            ))

    # High beta warning for expensive stocks
    _beta = nd.beta
    if (_beta is not None and math.isfinite(_beta) and _beta > 2.0
            and nd.pe_trailing is not None and nd.pe_trailing > 40):
        factors.append(StopFactor(
            name="High Volatility at Premium Valuation",
            description=(
                f"Beta {_beta:.1f} with P/E {nd.pe_trailing:.0f} — "
                "high-beta name at expensive valuation; "
                "drawdowns of 30-50% are historically common"
            ),
            severity="warning",
        ))

    # 6. Sharply deteriorating margins into deep negative
    margins = [x for x in nd.net_margin_annual[-3:] if not math.isnan(x)]
    if len(margins) == 3 and margins[2] < margins[1] < margins[0] and margins[2] < -10:
        factors.append(StopFactor(
            name="Deteriorating Margins",
            description=f"Net margin declining sharply to {margins[2]:.1f}%",
            severity="warning",
        ))

    # 7. Technical breakdown
    tech = blocks.get("technical")
    if tech is not None and tech.score < 3.0:
        factors.append(StopFactor(
            name="Technical Breakdown",
            description="Price in severe downtrend (score < 3/10)",
            severity="warning",
        ))

    # 8. Low liquidity — measured in dollar ADV (avg_volume × current_price)
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
# Trade recommendation
# ---------------------------------------------------------------------------

def _compute_trade_recommendation(
    nd: NormalisedData,
    overall_score: float,
    horizon: HorizonScores,
    stop_factors: list[StopFactor],
    blocks: dict[str, BlockScore],
    fv: Optional[FairValueResult],
) -> TradeRecommendation:
    """
    Convert scoring results into a concrete trading recommendation.

    Actions
    -------
    Accumulate            — qualifies fully at current price (market order)
    Accumulate on Pullback — score ≥ 60, but wait for better entry via limit order
    Avoid                 — score < 60 or critical stop factor
    """
    price = nd.current_price
    rationale: list[str] = []

    # ── Hard gates ───────────────────────────────────────────────────────
    has_critical = any(sf.severity == "critical" for sf in stop_factors)
    if price is None or not math.isfinite(price) or price <= 0:
        return TradeRecommendation(action="Avoid", entry_type="avoid",
                                   rationale=["Current price unavailable"])
    if has_critical:
        reasons = [sf.description for sf in stop_factors if sf.severity == "critical"]
        return TradeRecommendation(action="Avoid", entry_type="avoid",
                                   rationale=["Critical risk factor: " + r for r in reasons])
    if overall_score < 60:
        return TradeRecommendation(
            action="Avoid",
            entry_type="avoid",
            rationale=[f"Score {overall_score:.0f}/100 — below entry threshold (60)"],
        )

    # ── Horizon selection ────────────────────────────────────────────────
    # Prefer long when quality is high AND long score meaningfully exceeds medium.
    quality_score = blocks["quality"].score if "quality" in blocks else 5.0
    prefer_long = (
        horizon.long > horizon.medium + 4
        or (quality_score >= 8.0 and horizon.long >= horizon.medium)
    )
    if prefer_long:
        horizon_label = "long (1\u20135 years)"
        hold_months   = 24
    else:
        horizon_label = "medium (3\u201312 months)"
        hold_months   = 9

    # ── Target price (exit) ──────────────────────────────────────────────
    # Use analyst consensus median if available; fall back to fair_value × uplift.
    analyst_target = nd.analyst_target_median
    fv_price = fv.fair_value if fv is not None else None

    if analyst_target and math.isfinite(analyst_target) and analyst_target > price:
        target_price = analyst_target
        rationale.append(f"Exit target: analyst consensus ${analyst_target:.2f}")
    elif fv_price and math.isfinite(fv_price):
        uplift = 1.20 if prefer_long else 1.12
        target_price = max(fv_price, price * uplift)   # at least the uplift
        rationale.append(f"Exit target: fair value ${fv_price:.2f} + buffer")
    else:
        uplift = 1.20 if prefer_long else 1.12
        target_price = price * uplift
        rationale.append(f"Exit target: {(uplift-1)*100:.0f}% return target (no fair value available)")

    # Cap upside at 55% for long, 35% for medium (avoid implausible targets)
    max_upside = 1.55 if prefer_long else 1.35
    target_price = min(target_price, price * max_upside)

    # ── Stop loss ────────────────────────────────────────────────────────
    # Use MA200 as natural support when available; otherwise hard % stop.
    ma200: Optional[float] = None
    if len(nd.close_prices) >= 200:
        ma200 = sum(nd.close_prices[-200:]) / 200
    elif len(nd.close_prices) >= 50:
        ma200 = sum(nd.close_prices) / len(nd.close_prices)   # rough proxy

    hard_stop_pct = 0.87 if prefer_long else 0.89   # 11–13% hard stop
    if ma200 and math.isfinite(ma200) and ma200 < price * 0.99:
        # Stop just below MA200 (3% buffer) but no worse than hard stop
        stop_price = max(ma200 * 0.97, price * hard_stop_pct)
        rationale.append(f"Stop loss: below MA200 (${ma200:.2f}) at ${stop_price:.2f}")
    else:
        stop_price = price * hard_stop_pct
        rationale.append(f"Stop loss: {(1 - hard_stop_pct)*100:.0f}% hard stop at ${stop_price:.2f}")

    # ── Buy Now vs Buy on Limit ──────────────────────────────────────────
    # Buy Now: score ≥ 70 AND current price is within 5% above fair value
    # (or no fair value available and score ≥ 75 — model is confident)
    if fv_price and math.isfinite(fv_price):
        price_vs_fv_pct = (price - fv_price) / fv_price * 100  # positive = overpriced
    else:
        price_vs_fv_pct = 0.0  # unknown — treat as neutral

    # Entry quality gate — additional filters that must pass for BUY NOW.
    # Even a high-scoring stock fails the gate if:
    #   • technical score < 5: bearish structure (below MA200, downtrend)
    #   • risk score < 5: too risky at current price
    #   • price > 15% above MA50: extended / overheated short-term
    #   • R/R < 1.5: not enough upside relative to stop distance
    tech_score  = blocks["technical"].score  if "technical"  in blocks else 5.0
    risk_score  = blocks["risk"].score        if "risk"        in blocks else 5.0

    ma50: float | None = None
    if len(nd.close_prices) >= 50:
        ma50 = sum(nd.close_prices[-50:]) / 50

    price_above_ma50_pct = ((price / ma50) - 1) * 100 if ma50 else 0.0
    extended_vs_ma50 = price_above_ma50_pct > 15.0

    # Compute prospective R/R at current price (before deciding entry type)
    _rr_ratio = abs((target_price - price) / (price - stop_price)) if price > stop_price else 0.0

    gate_failures: list[str] = []
    if tech_score < 5.0:
        gate_failures.append(f"technical score {tech_score:.1f} < 5 (bearish structure or weak momentum)")
    if risk_score < 5.0:
        gate_failures.append(f"risk score {risk_score:.1f} < 5 (elevated risk at current price)")
    if extended_vs_ma50:
        gate_failures.append(f"price {price_above_ma50_pct:.0f}% above MA50 — extended, wait for pullback")
    if _rr_ratio < 1.5:
        gate_failures.append(f"R/R {_rr_ratio:.1f} : 1 below minimum 1.5 : 1")

    if overall_score >= 70 and price_vs_fv_pct <= 5.0 and not gate_failures:
        rationale.insert(0, f"Score {overall_score:.0f}/100 — qualifies for immediate entry")
        if fv_price:
            rationale.insert(1, f"Price ${price:.2f} within {abs(price_vs_fv_pct):.1f}% of fair value ${fv_price:.2f}")
        return TradeRecommendation(
            action="Accumulate",
            entry_type="market",
            horizon_label=horizon_label,
            hold_months=hold_months,
            limit_price=None,
            limit_wait_days=None,
            target_price=round(target_price, 2),
            stop_price=round(stop_price, 2),
            rationale=rationale,
        )

    # Buy on Limit: determine the limit price
    # If gate failed, explain why BUY NOW was downgraded
    if gate_failures and overall_score >= 70 and price_vs_fv_pct <= 5.0:
        rationale.insert(0, f"Score {overall_score:.0f}/100 — good, but entry gate not passed:")
        for gf in gate_failures:
            rationale.insert(1, f"  ✗ {gf}")
    if fv_price and math.isfinite(fv_price) and fv_price < price:
        # Stock is above fair value — wait for pullback to fair value
        limit_price = round(fv_price, 2)
        rationale.insert(0, f"Price ${price:.2f} is {price_vs_fv_pct:.1f}% above fair value — wait for pullback")
    else:
        # Stock is at/below fair value but score < 70 — buy on 3–5% dip
        dip_pct = 0.97 if overall_score >= 65 else 0.95
        limit_price = round(price * dip_pct, 2)
        rationale.insert(0, f"Score {overall_score:.0f}/100 — good but not exceptional; wait for {(1-dip_pct)*100:.0f}% dip")

    # If limit is more than 20% away the stock is materially overpriced vs fair
    # value — a realistic pullback of that magnitude requires a bear market or
    # earnings miss.  Treat this as Avoid rather than an impractical limit order.
    _pct_check = (price - limit_price) / price * 100
    if _pct_check > 20:
        return TradeRecommendation(
            action="Avoid",
            entry_type="avoid",
            rationale=[
                f"Price ${price:.2f} is {_pct_check:.0f}% above fair value ${limit_price:.2f}",
                "Stock materially overpriced — a limit order this far is impractical; revisit after re-rating",
            ],
        )

    pct_away = (price - limit_price) / price * 100

    # ── Dynamic wait_days based on ATR-14 ───────────────────────────────
    # Intuition: if the stock moves X% per day on average (ATR%), and the
    # limit is Y% away, it will reach the target in roughly Y/X trading days.
    # We add a buffer (×1.5) to account for directionality uncertainty, then
    # convert trading days → calendar days (×1.4, i.e. 5 trading = 7 calendar).
    #
    # Caps: min 2 calendar days (don't expire too fast), max 10 calendar days
    # (beyond that the market context has changed — re-run the analysis).
    # Fallback if ATR unavailable: use distance buckets (1%→2d, 3%→4d, 5%→6d).
    atr_pct = nd.atr_pct  # 14-day ATR as % of price, or None
    if atr_pct and atr_pct > 0:
        trading_days_est = (pct_away / atr_pct) * 1.5   # directional buffer
        calendar_days_est = trading_days_est * 1.4       # trading → calendar
        wait_days = max(2, min(10, round(calendar_days_est)))
        atr_note = f"ATR-14 {atr_pct:.2f}%/day → estimated {trading_days_est:.1f} trading days to fill"
    else:
        # Fallback: simple distance buckets
        if pct_away < 1.5:
            wait_days = 2
        elif pct_away < 3.0:
            wait_days = 4
        elif pct_away < 5.0:
            wait_days = 6
        else:
            wait_days = 8
        atr_note = "ATR unavailable — using distance-based estimate"

    # Adjust stop for limit entry
    stop_price_limit = max(limit_price * hard_stop_pct, stop_price * 0.97)
    if ma200 and math.isfinite(ma200) and ma200 < limit_price * 0.99:
        stop_price_limit = max(ma200 * 0.97, limit_price * hard_stop_pct)

    rationale.append(f"Limit {pct_away:.1f}% below current price — keep order {wait_days} calendar days ({atr_note})")

    return TradeRecommendation(
        action="Accumulate on Pullback",
        entry_type="limit",
        horizon_label=horizon_label,
        hold_months=hold_months,
        limit_price=limit_price,
        limit_wait_days=wait_days,
        target_price=round(target_price, 2),
        stop_price=round(stop_price_limit, 2),
        rationale=rationale,
    )


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

    # Apply sector-fit DQ override now that company_type is known
    _UNSUPPORTED_FIT = frozenset({CompanyType.FINANCIAL})
    _ADAPTED_FIT     = frozenset({CompanyType.CYCLICAL, CompanyType.PHARMA, CompanyType.TURNAROUND})
    if cr.company_type in _UNSUPPORTED_FIT:
        nd.dq_sector_fit = "unsupported"
        nd.dq_cashflow   = "sector_n_a"
    elif cr.company_type in _ADAPTED_FIT:
        nd.dq_sector_fit = "adapted"
    else:
        nd.dq_sector_fit = "native"

    blocks: dict[str, BlockScore] = {
        "quality":   score_quality(nd, bm, cr.company_type),
        "valuation": score_valuation(nd, bm, cr.company_type),
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

    trade_rec = _compute_trade_recommendation(
        nd=nd,
        overall_score=overall,
        horizon=HorizonScores(short=short, medium=medium, long=long_),
        stop_factors=stop_factors,
        blocks=blocks,
        fv=fair_value,
    )

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
        trade_rec=trade_rec,
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
