"""
Rule-based company-type classifier.

Determines which Benchmark profile to apply based on observable signals.
The rules are deliberately transparent so they can be audited and tuned.

Priority order (first matching rule wins):
  1. Sector shortcut  (Financial, Pharma)
  2. Dividend + stability → Dividend/Defensive
  3. High-growth + high-margin → Hypergrowth Tech
  4. Mature tech profile
  5. Cyclical sectors
  6. Turnaround (negative or near-zero margins, recovering)
  7. Other

TODO: Replace or augment with a small trained classifier (e.g. XGBoost)
      once we have labelled historical data.  The rule-based version serves
      as a strong baseline and explainability layer.
"""

from __future__ import annotations

from src.data.models.stock_data import StockData


# Sectors as reported by yfinance
_FINANCIAL_SECTORS  = {"Financial Services", "Financials"}
_HEALTHCARE_SECTORS = {"Healthcare"}
_ENERGY_SECTORS     = {"Energy", "Basic Materials", "Materials"}
_INDUSTRIAL_SECTORS = {"Industrials"}
_DEFENSIVE_SECTORS  = {"Consumer Defensive", "Utilities", "Real Estate"}
_TECH_SECTORS       = {"Technology", "Communication Services"}


def classify(sd: StockData) -> str:
    """
    Return a stock_type key matching BENCHMARKS registry.
    Sets sd.stock_type in-place and returns it.
    """
    stock_type = _classify(sd)
    sd.stock_type = stock_type
    return stock_type


def _classify(sd: StockData) -> str:
    sector  = sd.sector or ""
    rev_g   = sd.revenue_growth_yoy
    gm      = sd.gross_margin
    op_m    = sd.operating_margin
    d_yield = sd.dividend_yield or 0.0
    payout  = sd.payout_ratio   or 0.0
    dte     = sd.debt_to_equity

    # 1. Financial sector
    if sector in _FINANCIAL_SECTORS:
        return "financial"

    # 2. Healthcare / Pharma
    if sector in _HEALTHCARE_SECTORS:
        return "pharma_healthcare"

    # 3. Dividend / Defensive
    #    Stable dividend, low growth, low-volatility sector
    if sector in _DEFENSIVE_SECTORS:
        return "dividend_defensive"
    if d_yield >= 0.025 and (payout or 0) < 0.85 and (rev_g or 0) < 0.10:
        return "dividend_defensive"

    # 4. Cyclical
    if sector in _ENERGY_SECTORS or sector in _INDUSTRIAL_SECTORS:
        return "cyclical"

    # 5. Tech — Hypergrowth
    #    Revenue growth ≥ 20%, gross margin ≥ 50%
    if sector in _TECH_SECTORS:
        if _val(rev_g) >= 0.20 and _val(gm) >= 0.50:
            return "hypergrowth_tech"
        if _val(rev_g) >= 0.10 and _val(gm) >= 0.40:
            return "mature_tech"
        return "mature_tech"

    # 6. Non-sector hypergrowth (e.g. SHOP, SQ outside pure tech label)
    if _val(rev_g) >= 0.25 and _val(gm) >= 0.50:
        return "hypergrowth_tech"

    # 7. Turnaround signal: operating margin negative but revenue still positive
    if op_m is not None and op_m < 0.0:
        return "turnaround"
    if dte is not None and dte > 2.5:
        return "turnaround"

    return "other"


def _val(x, default: float = 0.0) -> float:
    return x if x is not None else default
