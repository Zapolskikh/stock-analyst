"""
Offline data snapshot builder.

Fetches raw SEC EDGAR + yfinance data for a set of tickers and saves
everything to data/offline/{TICKER}/ as plain files (JSON + Parquet).

Usage:
    python scripts/fetch_offline_data.py

Output structure:
    data/offline/
        SPY_ohlcv.parquet           ← shared benchmark prices
        {TICKER}/
            info.json               ← full yfinance .info dict
            ohlcv.parquet           ← 5-year daily OHLCV
            sec_{metric}.parquet    ← one file per SEC EDGAR concept
            sec_raw_facts.json      ← complete raw XBRL JSON from SEC

To use offline (in engine or tests), load from these files instead of
calling the live fetchers.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# Ensure project root is on path when running as a script
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.price import fetch_info, fetch_ohlcv  # noqa: E402
from src.data.sec_edgar import _fetch_raw_facts, fetch_fundamentals, get_cik  # noqa: E402

# ---------------------------------------------------------------------------
# Tickers: 20 stocks covering 8 sectors and all CompanyType categories
# ---------------------------------------------------------------------------
TICKERS = [
    # HYPERGROWTH_TECH
    "NVDA", "TSLA",
    # MATURE_TECH
    "MSFT", "AAPL", "ORCL",
    # PHARMA
    "LLY", "ABBV", "PFE",
    # CYCLICAL
    "HD", "GM",
    # DIVIDEND_DEFENSIVE
    "KO", "PG", "JNJ",
    # FINANCIAL
    "JPM", "BAC", "GS",
    # TURNAROUND
    "INTC", "NKE",
    # ENERGY / OTHER
    "XOM", "CVX",
]

OUT_DIR = ROOT / "data" / "offline"


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


def fetch_ticker(ticker: str) -> None:
    out = OUT_DIR / ticker
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"  {ticker}")
    print(f"{'='*55}")

    # ── 1. yfinance info ──────────────────────────────────────
    print("  [1/3] yfinance info ...", end=" ", flush=True)
    try:
        info = fetch_info(ticker)
        save_json(info, out / "info.json")
        print(f"OK  (keys={len(info)})")
    except Exception as exc:
        print(f"ERROR: {exc}")

    # ── 2. OHLCV 5-year ──────────────────────────────────────
    print("  [2/4] OHLCV 5y ...", end=" ", flush=True)
    try:
        df_price = fetch_ohlcv(ticker, period="5y")
        save_parquet(df_price, out / "ohlcv.parquet")
        print(f"OK  ({len(df_price)} rows)")
    except Exception as exc:
        print(f"ERROR: {exc}")

    # ── 2b. Split history ───────────────────────────────────────────
    print("  [3/4] Split history ...", end=" ", flush=True)
    try:
        t_obj = yf.Ticker(ticker)
        splits = t_obj.splits  # pd.Series: date → ratio
        if splits is not None and not splits.empty:
            df_splits = splits.reset_index()
            df_splits.columns = ["date", "ratio"]
            df_splits["date"] = pd.to_datetime(df_splits["date"]).dt.tz_localize(None)
            save_parquet(df_splits, out / "yf_splits.parquet")
            print(f"OK  ({len(df_splits)} splits: {df_splits['ratio'].tolist()})")
        else:
            # Save empty placeholder so load_offline knows the file was fetched
            df_splits = pd.DataFrame(columns=["date", "ratio"])
            save_parquet(df_splits, out / "yf_splits.parquet")
            print("OK  (no splits on record)")
    except Exception as exc:
        print(f"ERROR: {exc}")

    # ── 3. SEC EDGAR ──────────────────────────────────────────
    print("  [4/4] SEC EDGAR fundamentals ...", end=" ", flush=True)
    try:
        # Save parsed concepts (one parquet per metric)
        fundamentals = fetch_fundamentals(ticker)
        for metric, df in fundamentals.items():
            save_parquet(df, out / f"sec_{metric}.parquet")

        # Save the complete raw XBRL JSON (all facts, unfiltered)
        cik = get_cik(ticker)
        raw_facts = _fetch_raw_facts(cik)
        save_json(raw_facts, out / "sec_raw_facts.json")

        size_kb = (out / "sec_raw_facts.json").stat().st_size // 1024
        print(f"OK  ({len(fundamentals)} metrics, raw={size_kb} KB)")
    except Exception as exc:
        print(f"ERROR: {exc}")

    # SEC rate-limit: max 10 req/sec — be polite
    time.sleep(0.5)


def fetch_spy() -> None:
    """Save SPY benchmark prices — shared across all tickers."""
    spy_path = OUT_DIR / "SPY_ohlcv.parquet"
    print("\n  SPY benchmark ...", end=" ", flush=True)
    try:
        df = fetch_ohlcv("SPY", period="5y")
        save_parquet(df, spy_path)
        print(f"OK  ({len(df)} rows)")
    except Exception as exc:
        print(f"ERROR: {exc}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Fetch offline data snapshots")
    parser.add_argument("tickers", nargs="*", help="Tickers to fetch (default: all)")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers] if args.tickers else TICKERS

    print(f"Output directory: {OUT_DIR.resolve()}")
    print(f"Tickers ({len(tickers)}): {', '.join(tickers)}")
    print()

    fetch_spy()

    failed = []
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}]", end="")
        try:
            fetch_ticker(ticker)
        except Exception as exc:
            print(f"\n  !! Unexpected error for {ticker}: {exc}")
            failed.append(ticker)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "="*55)
    total_mb = sum(
        f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file()
    ) / 1_048_576
    print(f"Done. {len(tickers) - len(failed)}/{len(tickers)} tickers saved.")
    print(f"Total size: {total_mb:.1f} MB  →  {OUT_DIR.resolve()}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print()
    print("To use offline on another machine:")
    print("  1. Copy the data/offline/ folder to the target machine")
    print("  2. Run: python scripts/load_offline.py NVDA   (see that script)")


if __name__ == "__main__":
    main()
