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

# Types where PEG is unreliable and should be skipped:
#   FINANCIAL — bank/insurance EPS is dominated by provisions & one-offs,
#               not "growth" in the Lynch sense; P/E multiples don't scale
#               with EPS growth the way they do for operating companies.
_PEG_SKIP_TYPES: frozenset[CompanyType] = frozenset({
    CompanyType.FINANCIAL,
})

# Maximum EPS growth rate (%) used for P/E derivation in PEG model.
# Cyclicals and defensives are mean-reverting — a boom year shouldn't
# justify a 40%+ P/E; cap growth assumption conservatively by type.
_PEG_MAX_GROWTH: dict[CompanyType, float] = {
    CompanyType.HYPERGROWTH_TECH:   40.0,   # high growth is real
    CompanyType.MATURE_TECH:        30.0,
    CompanyType.PHARMA:             30.0,
    CompanyType.CYCLICAL:           18.0,   # boom years overstate trend
    CompanyType.DIVIDEND_DEFENSIVE: 15.0,   # slow-and-steady
    CompanyType.TURNAROUND:         25.0,
    CompanyType.OTHER:              25.0,
}

# Maximum fair P/E derived by PEG formula, capped by type.
# Prevents PEG from awarding growth-stock multiples to mature/defensive names.
# Example: MATURE_TECH with 30% growth → PEG gives 45x — unrealistic for IT services.
# These caps are based on long-run sector median forward P/E + modest premium.
_PEG_MAX_PE: dict[CompanyType, float] = {
    CompanyType.HYPERGROWTH_TECH:   60.0,   # growth premium allowed
    CompanyType.MATURE_TECH:        22.0,   # sector median ~13x + modest premium
    CompanyType.PHARMA:             25.0,
    CompanyType.CYCLICAL:           18.0,   # cyclicals rarely sustain high P/E
    CompanyType.DIVIDEND_DEFENSIVE: 20.0,
    CompanyType.TURNAROUND:         20.0,
    CompanyType.OTHER:              22.0,
}

# Base weights per model [dcf, analyst_consensus, peg_eps]
# Higher analyst weight for types where analyst coverage is rich & reliable
# Base weights [dcf, analyst_consensus, peg_eps] per company type.
# HYPERGROWTH_TECH: DCF is intentionally low (0.12) because:
#   - discount rate uncertainty dominates at high-beta
#   - 10-year terminal value assumptions are highly speculative for fast-movers
#   - analyst consensus + PEG are empirically better anchors for this cohort
# Even with declining beta fix, DCF at 20% still exerts too much drag on NVDA/TSLA.
_BASE_WEIGHTS: dict[CompanyType, tuple[float, float, float]] = {
    CompanyType.HYPERGROWTH_TECH:   (0.12, 0.58, 0.30),
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
    fair_value:       float                              # composite estimate ($)
    current_price:    float                              # market price ($)
    discount_pct:     float                              # positive = undervalued
    status:           str                                # Undervalued / Fairly Valued / Overvalued
    model_values:     dict[str, tuple[float, float]]     # name → (value, weight)
    assumptions:      list[str] = field(default_factory=list)
    model_spread_pct: float = 0.0                        # (max_model - min_model) / fair_value × 100
    dcf_range: tuple[float, float, float] | None = None  # (bear, base, bull) per-share estimates

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


def _last_positive(values: list[float]) -> float:
    """Like _last_valid but also skips zeros (e.g. SEC data with 0-filled gaps)."""
    for v in reversed(values):
        if math.isfinite(v) and v > 0:
            return v
    return float("nan")


def _recent_mean(values: list[float], n: int = 3) -> float | None:
    tail = [v for v in values[-n:] if math.isfinite(v)]
    return sum(tail) / len(tail) if tail else None


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _median(values: list[float]) -> float:
    """Median of a non-empty list of floats."""
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2


def _fcf_reliability(nd: NormalisedData) -> float:
    """
    DCF reliability score in [0.3, 1.0] based on FCF stability (CoV).

    CoV = std_dev / mean of last 5yr positive FCF values.
    Low CoV (stable, predictable FCF) → high score → DCF gets more weight.
    High CoV (volatile/cyclical FCF) → low score → DCF gets less weight.

    Examples:
      CTSH (CoV≈0.12) → reliability≈0.88 — steady IT services, DCF reliable
      CMI  (CoV≈0.56) → reliability≈0.44 — cyclical capex spikes, less reliable
      ETN  (CoV≈0.30) → reliability≈0.70 — moderate volatility
    """
    vals = [v for v in nd.fcf_annual[-5:] if math.isfinite(v) and v > 0]
    if len(vals) < 3:
        return 0.5  # insufficient history → neutral reliability
    mu = sum(vals) / len(vals)
    if mu <= 0:
        return 0.5
    variance = sum((v - mu) ** 2 for v in vals) / len(vals)
    cov = variance ** 0.5 / mu
    return max(0.3, 1.0 - cov)  # CoV=0.0 → 1.0;  CoV≥0.7 → 0.3


# ---------------------------------------------------------------------------
# Model 1 — DCF on FCF
# ---------------------------------------------------------------------------

def _dcf_scenario(
    fcf_base: float,
    g1: float,
    discount_rate: float,
    shares: float,
    net_cash: float,
) -> float:
    """Compute one DCF scenario (bear / base / bull) given explicit inputs.
    Uses two-stage model: Stage 1 years 1-5 at g1, Stage 2 years 6-10 decays
    linearly from g1 → _TERMINAL_G, then Gordon-growth terminal value.
    Returns fair value per share.
    """
    pv_sum = 0.0
    fcf_t = fcf_base
    for t in range(1, 6):
        fcf_t *= (1 + g1)
        pv_sum += fcf_t / (1 + discount_rate) ** t
    for t in range(6, 11):
        blend = (t - 5) / 5
        g_t = g1 * (1 - blend) + _TERMINAL_G * blend
        fcf_t *= (1 + g_t)
        pv_sum += fcf_t / (1 + discount_rate) ** t
    tv = fcf_t * (1 + _TERMINAL_G) / (discount_rate - _TERMINAL_G)
    pv_tv = tv / (1 + discount_rate) ** 10
    fair_equity = pv_sum + pv_tv + net_cash
    return fair_equity / shares


def _model_dcf(
    nd: NormalisedData,
    company_type: CompanyType,
    assumptions: list[str],
) -> tuple[float | None, tuple[float, float, float] | None]:
    """
    Two-stage DCF discounting free cash flow.
    Returns (base_value_per_share, (bear, base, bull)) or (None, None) if data insufficient.
    Scenarios:
      Bear: g1 × 0.55, discount_rate × 1.10  (growth 45% below base, cost of capital 10% higher)
      Base: g1 × 1.00, discount_rate × 1.00
      Bull: g1 × 1.45, discount_rate × 0.92  (growth 45% above base, cost of capital 8% lower)
    """
    # Base FCF: median of last 5yr positive values (robust to outliers).
    # Median avoids distortions from cyclical peak years, acquisition-year FCF
    # drops, and single-year capex spikes (e.g. CMI: $279M→$2.75B; VRTX: -$790M).
    # Requires ≥ 3 positive FCF years; falls back to TTM otherwise.
    fcf_pos = [v for v in nd.fcf_annual[-5:] if math.isfinite(v) and v > 0]
    if len(fcf_pos) >= 3:
        fcf_base = _median(fcf_pos)
    else:
        fcf_base = nd.ttm_fcf
        if fcf_base is None or not math.isfinite(fcf_base):
            fcf_base = _last_valid(nd.fcf_annual)
    if fcf_base is None or not math.isfinite(fcf_base) or fcf_base <= 0:
        return None, None  # negative or missing FCF → model skipped

    # Shares outstanding — use last *positive* value; fall back to mkt_cap/price
    shares = _last_positive(nd.shares_outstanding_annual)
    if not math.isfinite(shares) or shares <= 0:
        # Fallback: estimate from market cap and current price
        if (
            nd.market_cap
            and math.isfinite(nd.market_cap)
            and nd.current_price
            and nd.current_price > 0
        ):
            shares = nd.market_cap / nd.current_price
        else:
            return None, None

    # Discount rate via CAPM.
    # For hypergrowth companies with elevated beta (e.g. NVDA β=2.2), a flat
    # CAPM → r=16.8% overpunishes: it assumes permanent high risk over 10 years.
    # In reality these companies converge toward market beta (~1.0) as they mature.
    # For HYPERGROWTH_TECH with beta > 1.5 we use a blended DCF beta:
    #   dcf_beta = (current_beta + 1.0) / 2  — midpoint on the path to market beta.
    # Example: NVDA β=2.2 → dcf_beta=1.6 → r=13.3% (vs punishing 16.8% flat CAPM).
    beta = nd.beta if (nd.beta is not None and math.isfinite(nd.beta)) else 1.0
    if company_type == CompanyType.HYPERGROWTH_TECH and beta > 1.5:
        dcf_beta = (beta + 1.0) / 2  # blend: current beta → market beta (1.0)
        assumptions.append(
            f"DCF beta: blended {beta:.1f} → 1.0 over horizon → dcf_beta={dcf_beta:.2f} "
            f"(hypergrowth maturity discount)"
        )
    else:
        dcf_beta = beta
    discount_rate = _clamp(_RISK_FREE + dcf_beta * _ERP, _MIN_DISCOUNT, _MAX_DISCOUNT)

    # FCF growth rate: endpoint CAGR over the positive FCF history.
    # CAGR (first → last positive value) is more robust than avg(YoY) because:
    #   - avg(YoY) is inflated by bounce-back years after a trough
    #     (AMT: trough year $1.82B → recovery years inflate avg to 29.8%, CAGR = 2.4%)
    #   - cyclical companies with $279M→$2.75B swings give avg(YoY) = 276% (CMI)
    #     while CAGR = 11.9% correctly reflects the underlying business growth
    # If only 1 positive year, fall back to revenue growth as proxy.
    fcf_vals = [v for v in nd.fcf_annual if math.isfinite(v) and v > 0]
    if len(fcf_vals) >= 2:
        n_years = len(fcf_vals) - 1
        g1_raw = (fcf_vals[-1] / fcf_vals[0]) ** (1 / n_years) - 1
    else:
        g1_raw_rev = _recent_mean(nd.revenue_growth_annual, 3)
        g1_raw = (g1_raw_rev / 100) if g1_raw_rev is not None else 0.08

    # Fundamental cap: FCF growth can't sustainably exceed revenue growth × 1.3.
    # Margin expansion is real but bounded — prevents extrapolating above-trend FCF
    # for mature industrials (ETN: CAGR 22%, rev ~8% → cap 10.4%).
    # Not applied when revenue is shrinking (negative rev_g3 would invert the cap).
    rev_g3 = _recent_mean(nd.revenue_growth_annual, 3)
    if rev_g3 is not None and math.isfinite(rev_g3) and rev_g3 > 0:
        g1_raw = min(g1_raw, rev_g3 / 100 * 1.3)

    g1 = _clamp(g1_raw, -0.05, 0.30)  # Stage-1 growth per year, hard caps

    # Net cash adjustment (cash adds value, debt subtracts)
    cash = _last_valid(nd.cash_annual)
    debt = _last_valid(nd.long_term_debt_annual)
    net_cash = 0.0
    if math.isfinite(cash):
        net_cash += cash
    if math.isfinite(debt):
        net_cash -= debt

    # ── Base scenario ──────────────────────────────────────────────────────
    base_val = _dcf_scenario(fcf_base, g1, discount_rate, shares, net_cash)

    # ── Bear / Bull scenarios ──────────────────────────────────────────────
    # Bear: growth 45% lower, discount rate 10% higher
    # Bull: growth 45% higher (still hard-capped at 30%), discount 8% lower
    g1_bear = _clamp(g1 * 0.55, -0.05, 0.30)
    g1_bull = _clamp(g1 * 1.45, -0.05, 0.30)
    r_bear  = _clamp(discount_rate * 1.10, _MIN_DISCOUNT, _MAX_DISCOUNT)
    r_bull  = _clamp(discount_rate * 0.92, _MIN_DISCOUNT, _MAX_DISCOUNT)
    bear_val = _dcf_scenario(fcf_base, g1_bear, r_bear, shares, net_cash)
    bull_val = _dcf_scenario(fcf_base, g1_bull, r_bull, shares, net_cash)

    dcf_range = (
        round(bear_val, 2) if bear_val > 0 else None,
        round(base_val, 2) if base_val > 0 else None,
        round(bull_val, 2) if bull_val > 0 else None,
    )

    base_label = "median" if len(fcf_pos) >= 3 else "TTM"
    assumptions.append(
        f"DCF: base_FCF={base_label} ${fcf_base/1e9:.2f}B, "
        f"g\u2081={g1*100:.1f}%/yr, r={discount_rate*100:.1f}%, "
        f"terminal_g={_TERMINAL_G*100:.0f}%, horizon=10yr"
    )
    assumptions.append(
        f"DCF range: Bear ${bear_val:.2f} / Base ${base_val:.2f} / Bull ${bull_val:.2f}"
    )

    if base_val <= 0:
        return None, None
    return base_val, (
        bear_val if bear_val > 0 else None,
        base_val,
        bull_val if bull_val > 0 else None,
    )


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

def _model_peg_eps(
    nd: NormalisedData,
    company_type: CompanyType,
    assumptions: list[str],
) -> float | None:
    """
    PEG formula: fair_price = forwardEps × (eps_growth_pct × PEG_target)

    Uses analyst consensus forwardEps from yfinance (not our own estimate).
    eps_growth derived from historical EPS series (or revenue growth as proxy).

    Peter Lynch: PEG=1.0 is fair; we use 1.5 (institutional conservative standard).
    Skipped for FINANCIAL types where EPS volatility makes PEG unreliable.
    """
    if company_type in _PEG_SKIP_TYPES:
        return None

    # Skip when forward EPS implies a large earnings inflection vs TTM EPS.
    # fwd_eps > 2.5 × ttm_eps means trailing EPS is depressed by GAAP charges
    # (M&A amortisation, impairments, restructuring) that don't reflect
    # ongoing earning power. In such cases:
    #   - Historical EPS growth series is contaminated by the distortion year
    #   - implied_g would be > 150% → already filtered, but then fallback to
    #     trailing EPS growth (near 0%) produces an absurdly low fair P/E
    # Better to skip PEG entirely and let DCF + analyst consensus carry the weight.
    # Example: ABBV fwd_eps $16.23 vs TTM EPS $2.88 → ratio 5.6× → skip.
    _fwd_for_check = nd.forward_eps
    _ttm_for_check = nd.ttm_eps_diluted
    if _ttm_for_check is None or not math.isfinite(_ttm_for_check):
        _ttm_for_check = _last_valid(nd.eps_diluted_annual)
    if (
        _fwd_for_check is not None
        and math.isfinite(_fwd_for_check)
        and _fwd_for_check > 0
        and math.isfinite(_ttm_for_check)
        and _ttm_for_check > 0
        and _fwd_for_check > _ttm_for_check * 2.5
    ):
        return None

    # Skip if TTM EPS is not meaningful relative to price.
    # |EPS_ttm / price| < 0.5% → P/E framework inapplicable: the company is
    # near breakeven or in earnings transition (base-effect). In such cases,
    # implied forward growth becomes astronomically large (e.g. DDOG: $0.46/$202
    # = 0.23% → implied growth 517%), making any growth-based P/E meaningless.
    _ttm_check = nd.ttm_eps_diluted
    if _ttm_check is None or not math.isfinite(_ttm_check):
        _ttm_check = _last_valid(nd.eps_diluted_annual)
    if (
        math.isfinite(_ttm_check)
        and nd.current_price
        and nd.current_price > 0
        and abs(_ttm_check) / nd.current_price < 0.005
    ):
        return None

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
        # EPS growth for fair P/E calculation.
        # Priority:
        #   1. Implied forward growth = (forwardEps / ttm_eps - 1)
        #      This is analyst-consensus-backed and forward-looking.
        #      Critical for companies in temporary earnings slowdown where
        #      trailing 3yr average is near zero but analysts see recovery.
        #   2. Trailing 3yr avg EPS growth (historical, from SEC EDGAR)
        #   3. Revenue growth proxy (last resort)
        # We take the BLEND: 50% implied + 50% trailing when both are available,
        # but if trailing ≤ 2% and implied > 5%, implied gets full weight —
        # avoids stale-trailing-data distorting the estimate.
        ttm_eps = nd.ttm_eps_diluted
        if ttm_eps is None or not math.isfinite(ttm_eps):
            ttm_eps = _last_valid(nd.eps_diluted_annual)

        implied_g: float | None = None
        if (
            math.isfinite(ttm_eps)
            and ttm_eps > 0
            and fwd_eps > ttm_eps * 0.5  # sanity: forwardEps shouldn't be < 50% TTM
        ):
            implied_g = (fwd_eps / ttm_eps - 1) * 100  # e.g. 18.2% for CTSH
            # Implied growth > 100% = base effect / earnings inflection, not trend.
            # Don't feed explosive base-effect numbers into P/E derivation.
            if implied_g > 100.0:
                implied_g = None

        trailing_g = _recent_mean(nd.eps_growth_annual, 3)
        if trailing_g is None:
            trailing_g = _recent_mean(nd.revenue_growth_annual, 3)

        if implied_g is not None and trailing_g is not None:
            if trailing_g <= 2.0 and implied_g > 5.0:
                # Trailing is stale/flat; analysts see recovery → use implied
                eps_g = implied_g
            else:
                # Normal blend: 50/50 implied vs trailing
                eps_g = implied_g * 0.5 + trailing_g * 0.5
        elif implied_g is not None:
            eps_g = implied_g
        elif trailing_g is not None:
            eps_g = trailing_g
        else:
            eps_g = 8.0  # generic fallback

        source = "yfinance consensus"

    if fwd_eps <= 0:
        return None

    # EPS growth clamped to [2%, type-specific max] for P/E derivation
    # Below 2% → growth stock formula breaks down
    # Above type max → cyclical/defensive boom years overstate long-run trend
    max_g = _PEG_MAX_GROWTH.get(company_type, 25.0)
    eps_g_for_pe = _clamp(eps_g, 2.0, max_g)

    # Fair P/E = growth_rate_pct × PEG_target, capped by type-specific max P/E
    # Prevents growth-stock multiples being applied to mature/defensive companies
    max_pe = _PEG_MAX_PE.get(company_type, 22.0)
    fair_pe = min(eps_g_for_pe * _PEG_TARGET, max_pe)

    fair_price = fwd_eps * fair_pe

    # Reject if estimate exceeds 4× current price — too speculative.
    # This handles cyclical/semiconductor peak-earnings cases where forwardEps
    # is elevated and PEG formula produces implausibly high targets.
    if nd.current_price and fair_price > nd.current_price * 4:
        return None

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
    val_dcf, dcf_range = _model_dcf(nd, company_type, assumptions)
    val_analyst, analyst_confidence = _model_analyst_target(nd, assumptions)
    val_peg = _model_peg_eps(nd, company_type, assumptions)

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
    else:
        # Scale DCF weight by FCF reliability (inverse of CoV).
        # Volatile FCF (cyclicals, acquisitive companies) → less reliable DCF.
        # Example: CTSH CoV≈0.12 → 0.88x;  CMI CoV≈0.56 → 0.44x;  ETN → 0.70x
        w_dcf = w_dcf * _fcf_reliability(nd)
        # Cashflow anomaly: TTM OCF deviates sharply from history → DCF base FCF
        # is likely distorted by a one-off event. Halve the DCF weight so the
        # analyst target (or PEG) anchors the composite estimate instead.
        # Example: KO TTM OCF = −$2.5B vs 5yr avg $9.8B → DCF weight ×0.5
        if nd.cashflow_anomaly:
            w_dcf *= 0.5
            assumptions.append(
                "DCF weight halved — cashflow anomaly detected (TTM OCF diverges from history)"
            )
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

    # Context-aware analyst cap.
    # The cap is type- and model-count-aware:
    #
    #   3 models (DCF + Analyst + PEG): cap 40% — two quant models exist,
    #     analyst should be a minority voice.
    #
    #   2 models + HYPERGROWTH_TECH: cap 72% — DCF is highly unreliable for
    #     high-beta hypergrowth over a 10-year horizon; 50+ analysts covering
    #     the stock provide a far better anchor than a speculative DCF.
    #     Without a higher cap the excess redistribution forces DCF to 45%
    #     regardless of its 12% base weight — defeating the intent.
    #
    #   2 models + other types: cap 55% — DCF gets reasonable anchor role.
    models_available = sum(1 for v in [val_dcf, val_analyst, val_peg] if v is not None)
    if models_available >= 3:
        _ANALYST_MAX_WEIGHT = 0.40
    elif company_type == CompanyType.HYPERGROWTH_TECH:
        _ANALYST_MAX_WEIGHT = 0.72
    else:
        _ANALYST_MAX_WEIGHT = 0.55
    if w_analyst > _ANALYST_MAX_WEIGHT:
        excess = w_analyst - _ANALYST_MAX_WEIGHT
        w_analyst = _ANALYST_MAX_WEIGHT
        other_sum = w_dcf + w_peg
        if other_sum > 0:
            w_dcf += excess * (w_dcf / other_sum)
            w_peg += excess * (w_peg / other_sum)
        # If no other models: analyst keeps full weight (already handled above)

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

    # Model spread: how much do the individual estimates diverge from each other?
    # Expressed as (max - min) / composite × 100.  High spread (>60%) means the
    # models disagree significantly — a signal for the AI layer to investigate.
    raw_vals = [v for v, _ in model_values.values()]
    if len(raw_vals) >= 2 and fv > 0:
        model_spread_pct = round((max(raw_vals) - min(raw_vals)) / fv * 100, 1)
    else:
        model_spread_pct = 0.0

    return FairValueResult(
        fair_value=round(fv, 2),
        current_price=round(price, 2),
        discount_pct=round(discount_pct, 1),
        status=status,
        model_values=model_values,
        assumptions=assumptions,
        model_spread_pct=model_spread_pct,
        dcf_range=dcf_range,
    )
