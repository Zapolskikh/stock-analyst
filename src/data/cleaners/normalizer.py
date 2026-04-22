"""
Transform raw yfinance output → normalised StockData.

Handles:
  - extracting fields from info dict
  - computing FCF, ROIC from financial statement tables
  - computing technical fields from price history
  - filling sector_median placeholders (TODO: real sector medians)
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Optional

from src.data.models.stock_data import StockData


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def clean(ticker: str, raw: dict) -> StockData:
    info:       dict         = raw.get("info", {})
    history:    pd.DataFrame = raw.get("history", pd.DataFrame())
    financials: dict         = raw.get("financials", {})

    sd = StockData(ticker=ticker.upper())

    # -- Identity
    sd.name     = info.get("longName", "")
    sd.sector   = info.get("sector",   "Unknown")
    sd.industry = info.get("industry", "Unknown")

    # -- Price & market
    sd.price      = _f(info.get("currentPrice") or info.get("regularMarketPrice"))
    sd.market_cap = _f(info.get("marketCap"))

    # -- Growth
    sd.revenue_growth_yoy = _f(info.get("revenueGrowth"))   # already fraction
    sd.eps_growth_yoy     = _f(info.get("earningsGrowth"))

    # -- Margins
    sd.gross_margin     = _f(info.get("grossMargins"))
    sd.operating_margin = _f(info.get("operatingMargins"))
    sd.net_margin       = _f(info.get("profitMargins"))

    # -- Returns
    sd.roe  = _f(info.get("returnOnEquity"))
    sd.roic = _compute_roic(financials)     # derived

    # -- FCF
    sd.fcf        = _compute_fcf(financials)
    sd.fcf_margin = _safe_div(sd.fcf, _f(info.get("totalRevenue")))
    sd.fcf_yield  = _safe_div(sd.fcf, sd.market_cap)

    # -- Valuation
    sd.pe         = _f(info.get("trailingPE"))
    sd.forward_pe = _f(info.get("forwardPE"))
    sd.ev_ebitda  = _f(info.get("enterpriseToEbitda"))
    sd.ps         = _f(info.get("priceToSalesTrailing12Months"))
    sd.peg        = _f(info.get("pegRatio"))
    sd.p_fcf      = _safe_div(sd.market_cap, sd.fcf) if sd.fcf and sd.fcf > 0 else None

    # -- Debt
    sd.debt_to_equity   = _f(info.get("debtToEquity"))       # already ratio (×100 in yf)
    if sd.debt_to_equity:
        sd.debt_to_equity /= 100                             # normalise to plain ratio
    sd.interest_coverage = _compute_interest_coverage(financials)

    # -- Dividends
    sd.dividend_yield = _f(info.get("dividendYield"))
    sd.payout_ratio   = _f(info.get("payoutRatio"))

    # -- Technical (from price history)
    if not history.empty:
        _fill_technical(sd, history)

    # -- Volatility
    sd.beta = _f(info.get("beta"))

    return sd


# ---------------------------------------------------------------------------
# Technical calculations
# ---------------------------------------------------------------------------

def _fill_technical(sd: StockData, history: pd.DataFrame) -> None:
    closes = history["Close"].dropna()
    if len(closes) < 20:
        return

    price = closes.iloc[-1]
    sd.price = sd.price or float(price)

    # Moving averages
    if len(closes) >= 50:
        ma50 = closes.rolling(50).mean().iloc[-1]
        sd.price_vs_50ma = (price - ma50) / ma50

    if len(closes) >= 200:
        ma200 = closes.rolling(200).mean().iloc[-1]
        sd.price_vs_200ma = (price - ma200) / ma200

    # Momentum
    sd.momentum_3m  = _momentum(closes, 63)
    sd.momentum_6m  = _momentum(closes, 126)
    sd.momentum_12m = _momentum(closes, 252)

    # Max drawdown from 52-week high
    if len(closes) >= 252:
        window = closes.iloc[-252:]
    else:
        window = closes
    peak = window.max()
    sd.drawdown_52w = (price - peak) / peak   # negative number

    # Annualised volatility
    daily_ret = closes.pct_change().dropna()
    if len(daily_ret) > 20:
        sd.volatility_annualised = float(daily_ret.std() * math.sqrt(252))

    # TODO: relative strength vs SPY requires fetching SPY history separately


def _momentum(closes: pd.Series, window: int) -> Optional[float]:
    if len(closes) < window + 1:
        return None
    past  = closes.iloc[-(window + 1)]
    now   = closes.iloc[-1]
    return (now - past) / past


# ---------------------------------------------------------------------------
# Fundamental derivations
# ---------------------------------------------------------------------------

def _compute_fcf(financials: dict) -> Optional[float]:
    """Operating cash flow − capex (from cashflow statement)."""
    cf = financials.get("cashflow_annual")
    if cf is None or cf.empty:
        return None
    try:
        # yfinance row labels vary; try common names
        ocf  = _get_row(cf, ["Total Cash From Operating Activities",
                              "Operating Cash Flow"])
        capex = _get_row(cf, ["Capital Expenditures", "Capital Expenditure"])
        if ocf is None:
            return None
        capex_val = capex if capex is not None else 0.0
        # Use most recent column (index 0)
        return float(ocf.iloc[0]) + float(capex_val.iloc[0] if hasattr(capex_val, "iloc") else capex_val)
    except Exception:
        return None


def _compute_roic(financials: dict) -> Optional[float]:
    """NOPAT / (total equity + total debt).  Approximation."""
    try:
        inc = financials.get("income_annual")
        bal = financials.get("balance_annual")
        if inc is None or bal is None:
            return None

        ebit = _get_row(inc, ["Ebit", "EBIT", "Operating Income"])
        tax_rate = 0.21  # US blended approximation
        # TODO: use actual effective tax rate from income statement

        equity = _get_row(bal, ["Total Stockholder Equity", "Stockholders Equity",
                                 "Total Equity"])
        debt   = _get_row(bal, ["Long Term Debt", "Total Debt"])

        if ebit is None or equity is None:
            return None

        nopat      = float(ebit.iloc[0]) * (1 - tax_rate)
        ic_equity  = float(equity.iloc[0])
        ic_debt    = float(debt.iloc[0]) if debt is not None else 0.0
        invested_capital = ic_equity + ic_debt
        if invested_capital == 0:
            return None
        return nopat / invested_capital
    except Exception:
        return None


def _compute_interest_coverage(financials: dict) -> Optional[float]:
    try:
        inc = financials.get("income_annual")
        if inc is None:
            return None
        ebit     = _get_row(inc, ["Ebit", "EBIT", "Operating Income"])
        interest = _get_row(inc, ["Interest Expense"])
        if ebit is None or interest is None:
            return None
        int_val = abs(float(interest.iloc[0]))
        if int_val == 0:
            return None
        return float(ebit.iloc[0]) / int_val
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _f(val) -> Optional[float]:
    """Cast to float, return None if not possible."""
    try:
        v = float(val)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _get_row(df: pd.DataFrame, names: list[str]):
    """Try each row label; return the first match."""
    for name in names:
        if name in df.index:
            return df.loc[name]
    return None
