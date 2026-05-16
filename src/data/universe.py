"""Fetch ticker universes from live sources with file caching.

Supported universes:
  "sp500"              — S&P 500 (503 tickers) via Wikipedia, cached 24 h
  "russell1000"        — S&P 500 + S&P MidCap 400 (~903 tickers), cached 24 h
  "finviz_undervalued" — Finviz screener: Large Cap, P/E<15, P/B<2, ROE>15%,
                         cached 6 h.  NOT limited to S&P 500 (includes ADRs).
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
_CACHE_TTL_DAY  = 86_400   # 24 h  — index constituents (change rarely)
_CACHE_TTL_6H   = 21_600   # 6 h   — screener results (change more often)
_CACHE_TTL      = _CACHE_TTL_DAY  # default


# ── internal helpers ──────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    return _CACHE_DIR / f"universe_{name}.json"


def _load_cache(name: str, ttl: int = _CACHE_TTL) -> list[str] | None:
    p = _cache_path(name)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > ttl:
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list) and len(data) > 10:
            return data
    except Exception:
        pass
    return None


def _save_cache(name: str, tickers: list[str]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(name).write_text(
        json.dumps(tickers, ensure_ascii=False), encoding="utf-8"
    )


def _clean_tickers(raw: list[str]) -> list[str]:
    """Normalise a raw ticker list: strip, replace dots, drop non-equity rows."""
    out: list[str] = []
    for t in raw:
        t = str(t).strip().replace(".", "-")
        # keep only tickers that look real (1-7 alpha chars, optional hyphen+suffix)
        core = t.replace("-", "")
        if core.isalpha() and 1 <= len(t) <= 7:
            out.append(t)
    return out


# ── public fetchers ───────────────────────────────────────────────────────

def fetch_sp500(force_refresh: bool = False) -> list[str]:
    """Return S&P 500 tickers from Wikipedia (cached 24 h)."""
    if not force_refresh:
        cached = _load_cache("sp500")
        if cached:
            logger.debug("S&P 500: loaded %d tickers from cache", len(cached))
            return cached

    import pandas as pd
    import requests

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    logger.info("Fetching S&P 500 from Wikipedia…")
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()

    tables = pd.read_html(io.StringIO(resp.text))
    # first table is the constituents list; 'Symbol' column holds tickers
    tickers = _clean_tickers(tables[0]["Symbol"].tolist())

    if len(tickers) < 400:
        raise RuntimeError(
            f"S&P 500 fetch returned only {len(tickers)} tickers — unexpected format"
        )

    logger.info("S&P 500: fetched %d tickers", len(tickers))
    _save_cache("sp500", tickers)
    return tickers


def fetch_russell1000(force_refresh: bool = False) -> list[str]:
    """Return a Russell 1000 approximation: S&P 500 + S&P MidCap 400 (~900 stocks).

    Both lists are sourced from Wikipedia, so no authentication is needed.
    True Russell 1000 requires paid data from FTSE Russell; this approximation
    covers all the same large-cap names plus most mid-caps.
    """
    if not force_refresh:
        cached = _load_cache("russell1000")
        if cached:
            logger.debug("Russell 1000 (approx): loaded %d tickers from cache", len(cached))
            return cached

    import pandas as pd
    import requests

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    # --- S&P 400 mid-cap (the extension beyond S&P 500) ---
    logger.info("Fetching S&P 400 from Wikipedia…")
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        headers=_HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text), flavor="lxml")
    sp400 = _clean_tickers(tables[0]["Symbol"].tolist())

    # --- S&P 500 large-cap ---
    sp500 = fetch_sp500(force_refresh=force_refresh)

    # Combine, preserving order (large-caps first)
    seen: set[str] = set(sp500)
    extras = [t for t in sp400 if t not in seen]
    tickers = sp500 + extras

    if len(tickers) < 800:
        raise RuntimeError(
            f"Russell 1000 (approx) fetch returned only {len(tickers)} tickers"
        )

    logger.info(
        "Russell 1000 (approx): %d tickers (S&P 500: %d + S&P 400 extra: %d)",
        len(tickers), len(sp500), len(extras),
    )
    _save_cache("russell1000", tickers)
    return tickers


# ── Finviz undervalued screener ───────────────────────────────────────────

#: Default Finviz filter string for "undervalued quality large-caps".
#: Filters: Large Cap (>$10B), P/E < 15, P/B < 2, ROE > 15%.
#: Edit to taste — see https://finviz.com/screener.ashx for all filter codes.
FINVIZ_UNDERVALUED_FILTERS = "cap_largeover,fa_pe_u15,fa_pb_u2,fa_roe_o15"


def fetch_finviz_undervalued(
    filters: str = FINVIZ_UNDERVALUED_FILTERS,
    force_refresh: bool = False,
) -> list[str]:
    """Return tickers matching a Finviz screener query (cached 6 h).

    Paginates through all result pages automatically (20 results per page).
    The result is NOT limited to S&P 500 — includes any US-listed equity
    (NYSE, NASDAQ, AMEX, ADRs) that passes the filters.

    Default filters: Large Cap + P/E < 15 + P/B < 2 + ROE > 15%.
    Typical result size: 30–100 tickers depending on market conditions.
    """
    import re

    import requests
    from bs4 import BeautifulSoup

    cache_key = f"finviz_{filters.replace(',', '_')}"
    if not force_refresh:
        cached = _load_cache(cache_key, ttl=_CACHE_TTL_6H)
        if cached:
            logger.debug("Finviz undervalued: loaded %d tickers from cache", len(cached))
            return cached

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finviz.com/screener.ashx",
    }
    base_url = "https://finviz.com/screener.ashx"

    def _extract_tickers(soup: BeautifulSoup) -> list[str]:
        # Each row has a td with data-boxover-ticker attribute
        return list(dict.fromkeys(  # preserve order, deduplicate
            td["data-boxover-ticker"]
            for td in soup.select("[data-boxover-ticker]")
            if td.get("data-boxover-ticker")
        ))

    logger.info("Fetching Finviz undervalued screener (filters=%s)…", filters)

    # ── page 1 ────────────────────────────────────────────────────────────
    resp = requests.get(
        base_url, params={"v": "152", "f": filters}, headers=_HEADERS, timeout=30
    )
    resp.raise_for_status()
    soup1 = BeautifulSoup(resp.text, "lxml")

    all_tickers: list[str] = _extract_tickers(soup1)

    # determine total count from "#1 / 41 Total" text
    total_match = re.search(r"#\d+ / (\d+) Total", resp.text)
    total = int(total_match.group(1)) if total_match else len(all_tickers)

    # ── subsequent pages (r=21, r=41, …) ─────────────────────────────────
    for offset in range(21, total + 1, 20):
        resp = requests.get(
            base_url,
            params={"v": "152", "f": filters, "r": offset},
            headers=_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        page_tickers = _extract_tickers(BeautifulSoup(resp.text, "lxml"))
        for t in page_tickers:
            if t not in all_tickers:
                all_tickers.append(t)

    tickers = _clean_tickers(all_tickers)
    logger.info("Finviz undervalued: %d tickers (total reported: %d)", len(tickers), total)
    _save_cache(cache_key, tickers)
    return tickers


# ── dispatcher ────────────────────────────────────────────────────────────

_DISPLAY_NAMES: dict[str, str] = {
    "sp500":              "S&P 500",
    "russell1000":        "S&P 1500 (approx.)",
    "finviz_undervalued": "Finviz Undervalued",
}

_FETCHERS = {
    "sp500":              fetch_sp500,
    "russell1000":        fetch_russell1000,
    "finviz_undervalued": fetch_finviz_undervalued,
}


def fetch_universe(name: str, force_refresh: bool = False) -> tuple[list[str], str]:
    """Fetch constituents for *name*.

    Returns ``(tickers, display_label)`` where *display_label* is the
    human-readable source name (e.g. ``"S&P 500"``).

    Raises ``ValueError`` for unknown names.
    """
    key = name.lower().replace(" ", "").replace("&", "")
    # normalise common aliases
    aliases = {
        "snp500": "sp500", "sp500": "sp500", "s&p500": "sp500",
        "russell1000": "russell1000", "r1000": "russell1000", "iwb": "russell1000",
        "finviz_undervalued": "finviz_undervalued",
        "finvizundervalued":  "finviz_undervalued",
        "undervalued":        "finviz_undervalued",
    }
    key = aliases.get(key, key)

    if key not in _FETCHERS:
        raise ValueError(
            f"Unknown universe {name!r}. Supported: {list(_DISPLAY_NAMES.keys())}"
        )

    label = _DISPLAY_NAMES[key]
    tickers = _FETCHERS[key](force_refresh=force_refresh)
    return tickers, label
