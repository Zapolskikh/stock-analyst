"""
Price data fetcher — yfinance wrapper.

Fetches OHLCV history, dividends, and splits for any Yahoo Finance ticker.
All returned indexes are timezone-naive for consistent downstream handling.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf


def _strip_tz(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return a timezone-naive copy of a DatetimeIndex (no-op if already naive)."""
    if getattr(idx, "tz", None) is not None:
        return idx.tz_localize(None)
    return idx


def fetch_ohlcv(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Return OHLCV DataFrame with a timezone-naive DatetimeIndex.

    Columns: Open, High, Low, Close, Volume
    *auto_adjust=True* means prices are already split- and dividend-adjusted.
    """
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data returned for {ticker!r}")
    df.index = _strip_tz(df.index)
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def fetch_dividends(ticker: str) -> pd.Series:
    """Return historical cash dividends as a Series (date → USD per share)."""
    s: pd.Series = yf.Ticker(ticker).dividends
    if not s.empty:
        s.index = _strip_tz(s.index)
    return s


def fetch_splits(ticker: str) -> pd.Series:
    """Return historical stock splits as a Series (date → ratio, e.g. 4.0 for 4-for-1)."""
    s: pd.Series = yf.Ticker(ticker).splits
    if not s.empty:
        s.index = _strip_tz(s.index)
    return s


def fetch_info(ticker: str) -> dict:
    """Return Yahoo Finance info dict (name, sector, market cap, P/E, beta, etc.)."""
    return yf.Ticker(ticker).info or {}
