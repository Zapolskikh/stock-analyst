#!/usr/bin/env python3
"""
Data collection CLI — fetch prices, SEC fundamentals, and generate charts.

Usage:
    python fetch.py NVDA
    python fetch.py NVDA AAPL MSFT

Output:
    output/<TICKER>/
        price_chart.html      candlestick + MA + RSI + MACD
        revenue.html          annual revenue & YoY growth
        profitability.html    revenue / gross profit / EBIT / net income
        margins.html          gross / operating / net margin %
        cashflow.html         operating CF / CapEx / free cash flow
        balance_sheet.html    equity + liabilities stack + LT debt
        eps.html              diluted & basic EPS
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.data.price import fetch_ohlcv
from src.data.sec_edgar import fetch_fundamentals, get_cik
from src.charts.price_chart import build_price_chart
from src.charts.fundamental_chart import build_fundamental_charts

_SEP = "─" * 56


def _process(ticker: str) -> None:
    ticker = ticker.upper()
    out_dir = Path("output") / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{_SEP}")
    print(f"  {ticker}")
    print(_SEP)

    # ------------------------------------------------------------------
    # 1. Price data (yfinance)
    # ------------------------------------------------------------------
    print("[ 1/3 ] Price data (yfinance) ...", end=" ", flush=True)
    try:
        df_price = fetch_ohlcv(ticker, period="5y")
        print(
            f"OK  "
            f"({len(df_price):,} bars · "
            f"{df_price.index[0].date()} → {df_price.index[-1].date()})"
        )
    except Exception as exc:
        print(f"FAILED — {exc}")
        return

    # ------------------------------------------------------------------
    # 2. Fundamental data (SEC EDGAR)
    # ------------------------------------------------------------------
    print("[ 2/3 ] SEC EDGAR fundamentals ...", end=" ", flush=True)
    fundamentals: dict = {}
    try:
        cik = get_cik(ticker)
        fundamentals = fetch_fundamentals(ticker)
        print(
            f"OK  "
            f"(CIK {cik} · {len(fundamentals)} metrics: "
            f"{', '.join(sorted(fundamentals))})"
        )
    except Exception as exc:
        print(f"SKIPPED — {exc}")

    # ------------------------------------------------------------------
    # 3. Charts
    # ------------------------------------------------------------------
    print("[ 3/3 ] Generating charts ...")

    price_path = out_dir / "price_chart.html"
    build_price_chart(df_price, ticker).write_html(str(price_path))
    print(f"        price_chart      →  {price_path}")

    if fundamentals:
        for name, path in build_fundamental_charts(fundamentals, ticker, out_dir):
            print(f"        {name:<20} →  {path}")


def main(tickers: list[str]) -> None:
    for t in tickers:
        _process(t)
    print(f"\n{_SEP}")
    print("  Done. Charts saved to output/<TICKER>/")
    print(_SEP)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python fetch.py TICKER [TICKER ...]")
        sys.exit(1)
    main(args)
