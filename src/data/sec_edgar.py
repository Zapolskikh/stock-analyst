"""
SEC EDGAR XBRL data fetcher.

Uses the free, unauthenticated SEC EDGAR JSON API:
  https://www.sec.gov/files/company_tickers.json         — ticker → CIK map
  https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json  — all XBRL facts

SEC rate-limit policy: ≤ 10 requests/second per IP.
Both responses are cached to disk (see src/data/cache.py).

Covers US-listed companies only (those that file with the SEC).
For international tickers fall back to yfinance .financials.
"""
from __future__ import annotations

import time

import requests
import pandas as pd

from src.data.cache import load_json, save_json, is_fresh

_BASE = "https://data.sec.gov"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# SEC requires a descriptive User-Agent string (organisation + contact e-mail).
_HEADERS = {"User-Agent": "stock-analyst research@stock-analyst.local"}

# ---------------------------------------------------------------------------
# XBRL concepts to extract
#   metric_name → ([candidate tag names in priority order], preferred unit)
# ---------------------------------------------------------------------------
CONCEPTS: dict[str, tuple[list[str], str]] = {
    "revenue": (
        [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
        ],
        "USD",
    ),
    "gross_profit":      (["GrossProfit"], "USD"),
    "operating_income":  (["OperatingIncomeLoss"], "USD"),
    "net_income":        (["NetIncomeLoss"], "USD"),
    "eps_diluted":       (["EarningsPerShareDiluted"], "USD/shares"),
    "eps_basic":         (["EarningsPerShareBasic"], "USD/shares"),
    "total_assets":      (["Assets"], "USD"),
    "total_liabilities": (["Liabilities"], "USD"),
    "equity": (
        [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        "USD",
    ),
    "operating_cf": (["NetCashProvidedByUsedInOperatingActivities"], "USD"),
    "capex":        (["PaymentsToAcquirePropertyPlantAndEquipment"], "USD"),
    "long_term_debt": (
        ["LongTermDebt", "LongTermDebtNoncurrent"],
        "USD",
    ),
    "shares_outstanding": (["CommonStockSharesOutstanding"], "shares"),
    "rd_expense":  (["ResearchAndDevelopmentExpense"], "USD"),
    "dividends_paid": (
        ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
        "USD",
    ),
    "cash": (
        [
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsAndShortTermInvestments",
        ],
        "USD",
    ),
    "inventory": (["InventoryNet"], "USD"),
}


# ---------------------------------------------------------------------------
# CIK lookup
# ---------------------------------------------------------------------------

def _load_ticker_map() -> dict[str, str]:
    """Return {TICKER: '0001234567'} from SEC bulk file (cached 7 days)."""
    key = "sec_ticker_map"
    if is_fresh(key, max_age_hours=24 * 7, suffix=".json"):
        cached = load_json(key)
        if cached:
            return cached

    resp = requests.get(_TICKERS_URL, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    raw: dict = resp.json()
    # raw = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    mapping = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in raw.values()}
    save_json(key, mapping)
    return mapping


def get_cik(ticker: str) -> str:
    """Resolve a ticker symbol to its 10-digit zero-padded SEC CIK string."""
    m = _load_ticker_map()
    cik = m.get(ticker.upper())
    if not cik:
        raise ValueError(
            f"CIK not found for {ticker!r}. "
            "Company may not file with the SEC (non-US or OTC)."
        )
    return cik


# ---------------------------------------------------------------------------
# Company facts download
# ---------------------------------------------------------------------------

def _fetch_raw_facts(cik: str) -> dict:
    """Download companyfacts JSON from SEC EDGAR (cached 24 hours)."""
    key = f"sec_facts_{cik}"
    if is_fresh(key, max_age_hours=24, suffix=".json"):
        cached = load_json(key)
        if cached:
            return cached

    url = f"{_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    time.sleep(0.15)  # stay well within the 10 req/s limit
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    save_json(key, data)
    return data


# ---------------------------------------------------------------------------
# Concept extraction
# ---------------------------------------------------------------------------

def _extract_concept(
    facts: dict,
    tag: str,
    taxonomy: str = "us-gaap",
    preferred_unit: str = "USD",
) -> pd.DataFrame:
    """
    Extract one XBRL concept from a companyfacts blob.

    Returns a DataFrame with columns:
        end, start, val, form, filed, accn

    Only 10-K and 10-Q filings are included.
    Duplicates for the same (end, form) period are resolved by keeping
    the most recently *filed* value.
    """
    try:
        units_dict: dict = facts["facts"][taxonomy][tag]["units"]
    except KeyError:
        return pd.DataFrame()

    unit = preferred_unit if preferred_unit in units_dict else next(iter(units_dict), None)
    if unit is None:
        return pd.DataFrame()

    rows = [
        {
            "end":   pd.to_datetime(e["end"]),
            "start": pd.to_datetime(e.get("start", e["end"])),
            "val":   e["val"],
            "form":  e.get("form", ""),
            "filed": e.get("filed"),
            "accn":  e.get("accn"),
        }
        for e in units_dict[unit]
        if e.get("form") in ("10-K", "10-Q")
    ]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["filed"] = pd.to_datetime(df["filed"])
    df = (
        df.sort_values("filed", ascending=False)
          .drop_duplicates(subset=["end", "form"], keep="first")
          .sort_values("end")
          .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_fundamentals(ticker: str) -> dict[str, pd.DataFrame]:
    """
    Fetch all key financial metrics for *ticker* from SEC EDGAR.

    Returns a dict  metric_name → DataFrame  where each DataFrame has columns:
        end, start, val, form, filed, accn

    Both annual (form == "10-K") and quarterly (form == "10-Q") rows are
    included — filter on df["form"] == "10-K" for annual trends.

    Raises ValueError if the ticker is not found in the SEC CIK map.
    """
    cik = get_cik(ticker)
    facts = _fetch_raw_facts(cik)

    result: dict[str, pd.DataFrame] = {}
    for metric, (tags, unit) in CONCEPTS.items():
        for tag in tags:
            df = _extract_concept(facts, tag, preferred_unit=unit)
            if not df.empty:
                result[metric] = df
                break  # first matching tag wins

    return result
