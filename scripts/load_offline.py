"""
Offline data loader.

Loads raw data that was previously saved by scripts/fetch_offline_data.py
— no network access needed.

Usage:
    python scripts/load_offline.py NVDA
    python scripts/load_offline.py JPM --show-info
    python scripts/load_offline.py MSFT --normalise

Can also be imported and used directly in debug sessions:

    from scripts.load_offline import load_ticker, load_spy, normalise_offline

    nd = normalise_offline("NVDA")
    result = analyse_nd(nd)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "offline"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_info(ticker: str) -> dict:
    """Load saved yfinance info dict."""
    path = OUT_DIR / ticker / "info.json"
    if not path.exists():
        raise FileNotFoundError(f"No offline info for {ticker!r}. Run fetch_offline_data.py first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_ohlcv(ticker: str) -> pd.DataFrame:
    """Load saved OHLCV price history."""
    path = OUT_DIR / ticker / "ohlcv.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No offline OHLCV for {ticker!r}.")
    return pd.read_parquet(path)


def load_spy() -> list[float]:
    """Load saved SPY closing prices (last 252 trading days)."""
    path = OUT_DIR / "SPY_ohlcv.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    return [float(v) for v in df["Close"].dropna().tail(252)]


def load_fundamentals(ticker: str) -> dict[str, pd.DataFrame]:
    """Load saved SEC EDGAR fundamentals (parsed, one DataFrame per metric)."""
    folder = OUT_DIR / ticker
    result: dict[str, pd.DataFrame] = {}
    for parquet_file in sorted(folder.glob("sec_*.parquet")):
        metric = parquet_file.stem[4:]  # strip "sec_" prefix
        result[metric] = pd.read_parquet(parquet_file)
    if not result:
        raise FileNotFoundError(
            f"No offline SEC data for {ticker!r}. "
            "Run fetch_offline_data.py first."
        )
    return result


def load_raw_facts(ticker: str) -> dict:
    """Load complete raw XBRL JSON from SEC (for deep inspection)."""
    path = OUT_DIR / ticker / "sec_raw_facts.json"
    if not path.exists():
        raise FileNotFoundError(f"No offline raw SEC facts for {ticker!r}.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_ticker(ticker: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """
    Load all offline data for a ticker.

    Returns (fundamentals, price_df, info) — same types as the live fetchers.
    """
    return load_fundamentals(ticker), load_ohlcv(ticker), load_info(ticker)


# ---------------------------------------------------------------------------
# High-level: normalise + classify offline
# ---------------------------------------------------------------------------

def normalise_offline(ticker: str):
    """
    Normalise a ticker using offline data only.

    Returns a NormalisedData object ready for scoring.
    """
    from src.data.normalizer import normalise

    fundamentals, price_df, info = load_ticker(ticker)
    spy = load_spy()
    return normalise(fundamentals, price_df, info, ticker=ticker, spy_prices=spy)


def analyse_offline(ticker: str):
    """
    Run the full analysis pipeline on offline data.

    Returns an AnalysisResult (same as engine.analyse()).
    """
    from src.engine.engine import analyse_nd

    nd = normalise_offline(ticker)
    return analyse_nd(nd)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _list_available() -> list[str]:
    return sorted(
        d.name for d in OUT_DIR.iterdir()
        if d.is_dir() and (d / "info.json").exists()
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect offline data snapshots")
    parser.add_argument("ticker", nargs="?", help="Ticker symbol (e.g. NVDA)")
    parser.add_argument("--list", action="store_true", help="List available tickers")
    parser.add_argument("--show-info", action="store_true", help="Print key yfinance info fields")
    parser.add_argument("--show-sec", action="store_true", help="Print available SEC metrics")
    parser.add_argument("--normalise", action="store_true", help="Run normalise() and print summary")
    parser.add_argument("--analyse", action="store_true", help="Run full analysis pipeline")
    args = parser.parse_args()

    if args.list or not args.ticker:
        available = _list_available()
        print(f"Available tickers ({len(available)}): {', '.join(available)}")
        return

    ticker = args.ticker.upper()

    if args.show_info:
        info = load_info(ticker)
        keys = ["shortName", "sector", "industry", "currentPrice", "marketCap",
                "trailingPE", "forwardPE", "forwardEps", "targetMedianPrice",
                "numberOfAnalystOpinions", "beta", "dividendYield"]
        print(f"\n{ticker} — yfinance info snapshot")
        print("-" * 40)
        for k in keys:
            v = info.get(k)
            if v is not None:
                print(f"  {k:<30} {v}")

    if args.show_sec:
        fund = load_fundamentals(ticker)
        print(f"\n{ticker} — SEC EDGAR metrics ({len(fund)} total)")
        print("-" * 40)
        for metric, df in fund.items():
            annual = df[df["form"] == "10-K"] if "form" in df.columns else df
            print(f"  {metric:<25} {len(df):>4} rows  ({len(annual)} annual)")

    if args.normalise:
        nd = normalise_offline(ticker)
        print(f"\n{ticker} — NormalisedData summary")
        print("-" * 40)
        print(f"  current_price:      ${nd.current_price}")
        print(f"  market_cap:         ${nd.market_cap:,.0f}" if nd.market_cap else "  market_cap: None")
        print(f"  ttm_revenue:        ${nd.ttm_revenue:,.0f}" if nd.ttm_revenue else "  ttm_revenue: None")
        print(f"  ttm_net_income:     ${nd.ttm_net_income:,.0f}" if nd.ttm_net_income else "  ttm_net_income: None")
        print(f"  ttm_fcf:            ${nd.ttm_fcf:,.0f}" if nd.ttm_fcf else "  ttm_fcf: None")
        print(f"  forward_eps:        {nd.forward_eps}")
        print(f"  analyst_target:     {nd.analyst_target_median}  (n={nd.analyst_count})")
        print(f"  revenue_annual:     {[f'{v/1e9:.1f}B' for v in nd.revenue_annual[-4:]]}")
        print(f"  eps_diluted_annual: {nd.eps_diluted_annual[-4:]}")

    if args.analyse:
        result = analyse_offline(ticker)
        from src.output.formatter import format_result
        print(format_result(result))

    if not any([args.show_info, args.show_sec, args.normalise, args.analyse]):
        # Default: brief summary
        info = load_info(ticker)
        fund = load_fundamentals(ticker)
        ohlcv = load_ohlcv(ticker)
        print(f"\n{ticker} — offline snapshot summary")
        print("-" * 40)
        print(f"  Company:      {info.get('shortName', 'N/A')}")
        print(f"  Sector:       {info.get('sector', 'N/A')}")
        print(f"  Price:        ${info.get('currentPrice', 'N/A')}")
        print(f"  OHLCV rows:   {len(ohlcv)}")
        print(f"  SEC metrics:  {len(fund)}")
        print(f"  Price range:  {ohlcv.index[0].date()} → {ohlcv.index[-1].date()}")


if __name__ == "__main__":
    main()
