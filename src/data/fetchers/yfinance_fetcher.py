"""
Fetch raw data for a ticker using yfinance.

yfinance gives us:
  - price history
  - key stats / info dict  (P/E, margins, beta, sector, …)
  - quarterly / annual financials (income stmt, balance sheet, cash flow)

Limitations:
  - Some fields (ROIC, FCF explicit) are not directly in yf.info
    and must be derived from financials tables.
  - For non-US tickers data can be sparse.
  - Rate limits: be respectful; add a small sleep between bulk requests.

TODO: add optional FMP (Financial Modeling Prep) fallback for missing
      fundamentals — especially ROIC, normalised FCF, forward EPS estimates.
TODO: add caching layer (pickle / sqlite) to avoid re-fetching on every run.
"""

from __future__ import annotations

import time
import yfinance as yf
import pandas as pd
from typing import Optional


def fetch_ticker(ticker: str, *, price_history_days: int = 730) -> dict:
    """
    Return a raw dict with all data yfinance can provide.

    Parameters
    ----------
    ticker : str
        E.g. 'NVDA', 'AAPL', 'MSFT'
    price_history_days : int
        How many calendar days of daily OHLCV to fetch.

    Returns
    -------
    dict with keys:
        'info'       – yf.Ticker.info  (company metadata & ratios)
        'history'    – pd.DataFrame    (OHLCV daily)
        'financials' – dict of DataFrames:
                         'income_annual', 'income_quarterly',
                         'balance_annual', 'cashflow_annual'
    """
    t = yf.Ticker(ticker)

    period = _days_to_period(price_history_days)
    history: pd.DataFrame = t.history(period=period)

    financials = {
        "income_annual":     _safe_df(t.financials),
        "income_quarterly":  _safe_df(t.quarterly_financials),
        "balance_annual":    _safe_df(t.balance_sheet),
        "cashflow_annual":   _safe_df(t.cashflow),
    }

    return {
        "info":       t.info or {},
        "history":    history,
        "financials": financials,
    }


def fetch_batch(tickers: list[str], delay_sec: float = 0.5) -> dict[str, dict]:
    """Fetch multiple tickers sequentially with a small delay."""
    results: dict[str, dict] = {}
    for ticker in tickers:
        results[ticker] = fetch_ticker(ticker)
        time.sleep(delay_sec)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _days_to_period(days: int) -> str:
    if days <= 30:   return "1mo"
    if days <= 90:   return "3mo"
    if days <= 180:  return "6mo"
    if days <= 365:  return "1y"
    if days <= 730:  return "2y"
    if days <= 1825: return "5y"
    return "max"


def _safe_df(obj) -> Optional[pd.DataFrame]:
    try:
        return obj if obj is not None and not obj.empty else None
    except Exception:
        return None
