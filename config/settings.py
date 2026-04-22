"""
Global configuration for the stock-analyst engine.

DATA SOURCES
------------
Primary : yfinance  (free, no key required)
Optional: Financial Modeling Prep — set FMP_API_KEY env variable for
          deeper fundamental data (income statement, cash flow, etc.)
Optional: Alpha Vantage  — set AV_API_KEY for intraday / extended data

TODO: evaluate which provider gives the best fundamental coverage
      for international tickers (non-US).  yfinance can be unreliable
      for some emerging-market symbols.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Load .env file if present (no-op when the file doesn't exist)
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Data provider keys  (loaded from environment / .env)
# ---------------------------------------------------------------------------
import os

FMP_API_KEY: str = os.getenv("FMP_API_KEY", "")
AV_API_KEY:  str = os.getenv("AV_API_KEY", "")

# ---------------------------------------------------------------------------
# Scoring weights  (MVP defaults — needs backtesting to optimise)
# ---------------------------------------------------------------------------
# TODO: run historical back-test to calibrate per-type weights
DEFAULT_WEIGHTS: dict[str, float] = {
    "quality":    0.30,
    "valuation":  0.25,
    "technical":  0.15,
    "risk":       0.20,
    "style_fit":  0.10,
}

# Horizon sub-weights override (relative importance within each horizon)
HORIZON_WEIGHTS: dict[str, dict[str, float]] = {
    "short": {
        "technical":  0.35,
        "risk":       0.25,
        "valuation":  0.20,
        "quality":    0.15,
        "style_fit":  0.05,
    },
    "medium": {
        "valuation":  0.30,
        "quality":    0.25,
        "technical":  0.20,
        "risk":       0.15,
        "style_fit":  0.10,
    },
    "long": {
        "quality":    0.40,
        "risk":       0.25,
        "valuation":  0.20,
        "style_fit":  0.10,
        "technical":  0.05,
    },
}

# ---------------------------------------------------------------------------
# Rating bands
# ---------------------------------------------------------------------------
RATING_BANDS: list[tuple[float, str]] = [
    (85, "Strong Candidate"),
    (70, "Good Candidate"),
    (55, "Neutral / Watchlist"),
    (40, "Weak"),
    (0,  "Avoid"),
]

# ---------------------------------------------------------------------------
# Data cache
# ---------------------------------------------------------------------------
CACHE_DIR: str = os.getenv("CACHE_DIR", "data/cache")
CACHE_TTL_HOURS: int = int(os.getenv("CACHE_TTL_HOURS", "24"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
import logging
LOG_LEVEL: int = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

# ---------------------------------------------------------------------------
# Historical window for technical analysis
# ---------------------------------------------------------------------------
PRICE_HISTORY_DAYS: int = 365 * 2  # 2 years
