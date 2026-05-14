"""
Fair Value estimation — 3-model composite (v2: максимально объективная).

Методология: средневзвешенное трёх независимых моделей.
Никаких субъективных мультипликаторов — только математически выводимые параметры.

Модели
------
1. DCF на FCF (Discounted Cash Flow)
   - Базовый FCF: TTM FCF → last annual FCF
   - Стадия 1 (годы 1–5): рост из исторических данных, ограничен [-5%, 30%]
   - Стадия 2 (годы 6–10): линейный переход к терминальному росту 3%
   - Ставка дисконтирования: CAPM = risk_free + beta × ERP (полностью выводимая)
   - Терминальная стоимость (Гордон): FCF₁₀ × (1+g) / (r − g)
   - На акцию: (Σ PV(FCF) + PV(TV) + cash_per_share − debt_per_share)

2. Analyst Consensus Target
   - Источник: yfinance targetMedianPrice (медиана → робастнее к outliers)
   - Вес пропорционален количеству аналитиков (больше аналитиков → выше уверенность)
   - Требуется минимум 3 аналитика; пропускается если данных нет
   - Это «crowd wisdom» десятков профессиональных аналитиков — наиболее объективная модель

3. PEG-EPS (Peter Lynch / Benjamin Graham)
   - Форвардный EPS: analyst forwardEps из yfinance (consensus, не наша оценка)
   - Fair P/E = EPS_growth% × PEG_target (стандарт: 1.5 — консервативнее Lynch 1.0)
   - Fair Price = forwardEps × fair_PE
   - Не требует субъективных мультипликаторов — fair_PE выводится из роста прибыли

Веса
----
Базовые веса [dcf, analyst, peg]: зависят от доступности данных и типа компании.
Если модель недоступна — веса перераспределяются между оставшимися.
Вес analyst масштабируется по числу аналитиков (confidence proxy).

Вывод
-----
FairValueResult:
  fair_value         — итоговая оценка ($)
  current_price      — текущая цена ($)
  discount_pct       — ((fair_value / price) - 1) × 100
  status             — "Undervalued" | "Fairly Valued" | "Overvalued"
  model_values       — dict model_name → (value, weight_used)
  assumptions        — список строк с ключевыми допущениями
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.classifier import CompanyType
from src.data.normalizer import NormalisedData


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RISK_FREE    = 0.045   # 10yr US Treasury yield (approximate)
_ERP          = 0.055   # Equity Risk Premium (historical average)
_TERMINAL_G   = 0.030   # Long-run perpetuity growth rate
_MIN_DISCOUNT = 0.080   # Floor on discount rate
_MAX_DISCOUNT = 0.180   # Cap on discount rate

# PEG target — Peter Lynch standard is 1.0, but 1.5 is more conservative
# and widely used by institutional analysts as "fair value" PEG
_PEG_TARGET = 1.5

# Minimum analysts required for analyst consensus model
_MIN_ANALYSTS = 3

# Base weights per model [dcf, analyst_consensus, peg_eps]
# Higher analyst weight for types where analyst coverage is rich & reliable
_BASE_WEIGHTS: dict[CompanyType, tuple[float, float, float]] = {
    CompanyType.HYPERGROWTH_TECH:   (0.20, 0.50, 0.30),
    CompanyType.MATURE_TECH:        (0.35, 0.40, 0.25),
    CompanyType.PHARMA:             (0.30, 0.45, 0.25),
    CompanyType.CYCLICAL:           (0.40, 0.35, 0.25),
    CompanyType.DIVIDEND_DEFENSIVE: (0.40, 0.35, 0.25),
    CompanyType.FINANCIAL:          (0.20, 0.55, 0.25),
    CompanyType.TURNAROUND:         (0.20, 0.55, 0.25),
    CompanyType.OTHER:              (0.30, 0.45, 0.25),
}

# Fair Value discount band (%)
_UNDERVALUED_THRESHOLD  =  15.0   # discount ≥ 15% → Undervalued
_OVERVALUED_THRESHOLD   = -15.0   # premium ≥ 15%  → Overvalued


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FairValueResult:
    fair_value:    float                              # composite estimate ($)
    current_price: float                              # market price ($)
    discount_pct:  float                              # positive = undervalued
    status:        str                                # Undervalued / Fairly Valued / Overvalued
    model_values:  dict[str, tuple[float, float]]     # name → (value, weight)
    assumptions:   list[str] = field(default_factory=list)

    @property
    def upside_str(self) -> str:
        """Human-readable upside/downside string, e.g. '+23.4% upside'."""
        sign = "+" if self.discount_pct >= 0 else ""
        label = "upside" if self.discount_pct >= 0 else "downside"
        return f"{sign}{self.discount_pct:.1f}% {label}"

    @property
    def status_icon(self) -> str:
        icons = {
            "Undervalued":   "🟢",
            "Fairly Valued": "🟡",
            "Overvalued":    "🔴",
        }
        return icons.get(self.status, "⚪")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _last_valid(values: list[float]) -> float:
    for v in reversed(values):
        if math.isfinite(v):
            return v
    return float("nan")


def _recent_mean(values: list[float], n: int = 3) -> float | None:
    tail = [v for v in values[-n:] if math.isfinite(v)]
    return sum(tail) / len(tail) if tail else None


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# Model 1 — DCF on FCF
# ---------------------------------------------------------------------------

def _model_dcf(nd: NormalisedData, assumptions: list[str]) -> float | None:
    """
    Two-stage DCF discounting free cash flow.
    Returns fair value per share, or None if data is insufficient.
    """
    # Base FCF
    fcf_base = nd.ttm_fcf
    if fcf_base is None or not math.isfinite(fcf_base):
        fcf_base = _last_valid(nd.fcf_annual)
    if not math.isfinite(fcf_base) or fcf_base <= 0:
        return None  # negative or missing FCF → model skipped

    # Shares outstanding
    shares = _last_valid(nd.shares_outstanding_annual)
    if not math.isfinite(shares) or shares <= 0:
        return None

    # Discount rate via CAPM
    beta = nd.beta if (nd.beta is not None and math.isfinite(nd.beta)) else 1.0
    discount_rate = _clamp(_RISK_FREE + beta * _ERP, _MIN_DISCOUNT, _MAX_DISCOUNT)

    # FCF growth rate: average of last 3y FCF growth → clamp
    fcf_vals = [v for v in nd.fcf_annual if math.isfinite(v) and v > 0]
    if len(fcf_vals) >= 2:
        yoy = [(fcf_vals[i] / fcf_vals[i - 1] - 1) for i in range(1, len(fcf_vals))]
        g1_raw = sum(yoy[-3:]) / len(yoy[-3:])
    else:
        # Fall back to revenue growth as proxy
        g1_raw_rev = _recent_mean(nd.revenue_growth_annual, 3)
        g1_raw = (g1_raw_rev / 100) if g1_raw_rev is not None else 0.08

    g1 = _clamp(g1_raw, -0.05, 0.30)  # Stage-1 growth per year

    # Stage 1: years 1–5
    pv_sum = 0.0
    fcf_t = fcf_base
    for t in range(1, 6):
        fcf_t *= (1 + g1)
        pv_sum += fcf_t / (1 + discount_rate) ** t

    # Stage 2: years 6–10, growth linearly decays to terminal_g
    for t in range(6, 11):
        blend = (t - 5) / 5   # 0 at t=6 → 1 at t=10
        g_t = g1 * (1 - blend) + _TERMINAL_G * blend
        fcf_t *= (1 + g_t)
        pv_sum += fcf_t / (1 + discount_rate) ** t

    # Terminal value (Gordon growth model, t=10)
    tv = fcf_t * (1 + _TERMINAL_G) / (discount_rate - _TERMINAL_G)
    pv_tv = tv / (1 + discount_rate) ** 10

    # Net cash adjustment (cash adds value, debt subtracts)
    cash = _last_valid(nd.cash_annual)
    debt = _last_valid(nd.long_term_debt_annual)
    net_cash = 0.0
    if math.isfinite(cash):
        net_cash += cash
    if math.isfinite(debt):
        net_cash -= debt

    fair_equity = pv_sum + pv_tv + net_cash
    fair_per_share = fair_equity / shares

    assumptions.append(
        f"DCF: g₁={g1*100:.1f}%/yr, r={discount_rate*100:.1f}%, "
        f"terminal_g={_TERMINAL_G*100:.0f}%, horizon=10yr"
    )
    return fair_per_share if fair_per_share > 0 else None


# ---------------------------------------------------------------------------
# Model 2 — Analyst Consensus Target Price
# ---------------------------------------------------------------------------

def _model_analyst_target(nd: NormalisedData, assumptions: list[str]) -> tuple[float | None, float]:
    """
    Use analyst consensus median price target from yfinance.
    Returns (value, confidence_weight_multiplier).
    Higher analyst count → higher confidence → higher weight multiplier.
    """
    target = nd.analyst_target_median
    if target is None or not math.isfinite(target) or target <= 0:
        target = nd.analyst_target_mean
    if target is None or not math.isfinite(target) or target <= 0:
        return None, 0.0

    count = nd.analyst_count or 0
    if count < _MIN_ANALYSTS:
        return None, 0.0

    # Confidence multiplier: scales 0.6→1.0→1.3 based on analyst count
    # < 5 analysts: 0.6x (thin coverage)
    # 5–15 analysts: 0.8–1.0x (normal)
    # > 15 analysts: up to 1.3x (broad institutional coverage)
    if count < 5:
        confidence = 0.60
    elif count <= 15:
        confidence = 0.80 + (count - 5) / 10 * 0.20   # 0.80 → 1.00
    else:
        confidence = min(1.0 + (count - 15) / 40 * 0.30, 1.30)  # 1.00 → 1.30

    assumptions.append(
        f"Analyst target: ${target:.2f} "
        f"(n={count} analysts, confidence={confidence:.2f}x)"
    )
    return target, confidence


# ---------------------------------------------------------------------------
# Model 3 — PEG-EPS (analyst forwardEps × growth-derived fair P/E)
# ---------------------------------------------------------------------------

def _model_peg_eps(nd: NormalisedData, assumptions: list[str]) -> float | None:
    """
    PEG formula: fair_price = forwardEps × (eps_growth_pct × PEG_target)
    
    Uses analyst consensus forwardEps from yfinance (not our own estimate).
    eps_growth derived from historical EPS series (or revenue growth as proxy).
    
    Peter Lynch: PEG=1.0 is fair; we use 1.5 (institutional conservative standard).
    """
    # Prefer analyst consensus forwardEps (most accurate — not our estimate)
    fwd_eps = nd.forward_eps
    if fwd_eps is None or not math.isfinite(fwd_eps) or fwd_eps <= 0:
        # Fallback: derive from TTM EPS + growth
        ttm_eps = nd.ttm_eps_diluted
        if ttm_eps is None or not math.isfinite(ttm_eps):
            ttm_eps = _last_valid(nd.eps_diluted_annual)
        if not math.isfinite(ttm_eps) or ttm_eps <= 0:
            return None
        eps_g = _recent_mean(nd.eps_growth_annual, 3)
        if eps_g is None:
            eps_g = _recent_mean(nd.revenue_growth_annual, 3) or 8.0
        eps_g_clamped = _clamp(eps_g / 100, 0.0, 0.40)
        fwd_eps = ttm_eps * (1 + eps_g_clamped)
        source = "derived"
    else:
        # EPS growth for fair P/E calculation (still need growth rate)
        eps_g = _recent_mean(nd.eps_growth_annual, 3)
        if eps_g is None:
            eps_g = _recent_mean(nd.revenue_growth_annual, 3) or 8.0
        source = "yfinance consensus"

    if fwd_eps <= 0:
        return None

    # EPS growth clamped to [2%, 40%] for P/E derivation
    # Below 2% → growth stock formula breaks down; above 40% → unsustainable
    eps_g_for_pe = _clamp(eps_g, 2.0, 40.0)

    # Fair P/E = growth_rate_pct × PEG_target
    fair_pe = eps_g_for_pe * _PEG_TARGET

    fair_price = fwd_eps * fair_pe
    assumptions.append(
        f"PEG-EPS: fwd_EPS=${fwd_eps:.2f} ({source}), "
        f"g={eps_g_for_pe:.1f}%, PEG={_PEG_TARGET}, fair_P/E={fair_pe:.1f}x"
    )
    return fair_price if fair_price > 0 else None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_fair_value(
    nd: NormalisedData,
    company_type: CompanyType,
) -> FairValueResult | None:
    """
    Compute composite Fair Value estimate (v2 — objective, analyst-consensus driven).

    Returns None when current_price is unavailable or no model produces
    a valid estimate.
    """
    price = nd.current_price
    if price is None or not math.isfinite(price) or price <= 0:
        return None

    assumptions: list[str] = []

    # ── Run the three models ──────────────────────────────────────────────
    val_dcf = _model_dcf(nd, assumptions)
    val_analyst, analyst_confidence = _model_analyst_target(nd, assumptions)
    val_peg = _model_peg_eps(nd, assumptions)

    # Sanity cap: clip each estimate to [0.1×price, 10×price]
    lo, hi = price * 0.10, price * 10.0

    def _sanitize(v: float | None) -> float | None:
        if v is None or not math.isfinite(v):
            return None
        if v <= 0 or v < lo or v > hi:
            return None
        return v

    val_dcf = _sanitize(val_dcf)
    val_analyst = _sanitize(val_analyst)
    val_peg = _sanitize(val_peg)

    # ── Compute weights ───────────────────────────────────────────────────
    w_dcf, w_analyst, w_peg = _BASE_WEIGHTS.get(company_type, (0.30, 0.45, 0.25))

    if val_dcf is None:
        w_dcf = 0.0
    if val_analyst is None:
        w_analyst = 0.0
    else:
        # Scale analyst weight by confidence multiplier (clamp within [0.5w, 1.5w])
        w_analyst = _clamp(w_analyst * analyst_confidence, w_analyst * 0.5, w_analyst * 1.5)
    if val_peg is None:
        w_peg = 0.0

    total_w = w_dcf + w_analyst + w_peg
    if total_w == 0:
        return None

    # Normalise weights
    w_dcf     /= total_w
    w_analyst /= total_w
    w_peg     /= total_w

    # Composite fair value
    fv = (
        (val_dcf or 0.0) * w_dcf +
        (val_analyst or 0.0) * w_analyst +
        (val_peg or 0.0) * w_peg
    )

    discount_pct = (fv / price - 1) * 100

    if discount_pct >= _UNDERVALUED_THRESHOLD:
        status = "Undervalued"
    elif discount_pct <= _OVERVALUED_THRESHOLD:
        status = "Overvalued"
    else:
        status = "Fairly Valued"

    model_values: dict[str, tuple[float, float]] = {}
    if val_dcf is not None:
        model_values["DCF"] = (val_dcf, w_dcf)
    if val_analyst is not None:
        model_values["Analyst"] = (val_analyst, w_analyst)
    if val_peg is not None:
        model_values["PEG-EPS"] = (val_peg, w_peg)

    return FairValueResult(
        fair_value=round(fv, 2),
        current_price=round(price, 2),
        discount_pct=round(discount_pct, 1),
        status=status,
        model_values=model_values,
        assumptions=assumptions,
    )
