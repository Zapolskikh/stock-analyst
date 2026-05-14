"""
Data normalisation layer — Шаг 2 алгоритма.

Принимает "сырые" данные из sec_edgar / price и возвращает
структуру NormalisedData, готовую к классификации и скорингу.

Что делает:
  1. Извлекает последовательность годовых значений (10-K) для каждой метрики.
  2. Вычисляет производные показатели:
       - margins (gross / operating / net)
       - FCF = operating_cf − capex
       - revenue_growth (YoY %)
       - eps_growth (YoY %)
       - debt_to_equity
       - roe  = net_income / equity
       - roa  = net_income / total_assets
       - current_price, market_cap, pe_ratio, forward_pe, beta, sector
  3. Определяет «здоровье» данных: сколько лет истории, есть ли пропуски.
  4. Не падает при частичных данных — отсутствующие метрики → None / NaN.

Публичный интерфейс
-------------------
    from src.data.normalizer import normalise

    nd = normalise(fundamentals, price_df, info)
    nd.revenue_annual          # list[float]  — выручка по годам, USD
    nd.gross_margin_annual     # list[float]  — gross margin %, по годам
    nd.fcf_annual              # list[float]  — FCF по годам, USD
    nd.revenue_growth_annual   # list[float]  — YoY % (первый элемент = NaN)
    nd.roe_annual              # list[float]
    nd.years                   # list[int]    — список фискальных лет
    nd.current_price           # float | None
    nd.market_cap              # float | None
    nd.pe_trailing             # float | None
    nd.pe_forward              # float | None
    nd.beta                    # float | None
    nd.sector                  # str | None
    nd.years_of_history        # int
    nd.data_quality            # "good" | "partial" | "poor"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class NormalisedData:
    ticker: str

    # --- Annual series (aligned by fiscal year, oldest → newest) -----------
    years: list[int] = field(default_factory=list)

    revenue_annual: list[float] = field(default_factory=list)
    gross_profit_annual: list[float] = field(default_factory=list)
    operating_income_annual: list[float] = field(default_factory=list)
    net_income_annual: list[float] = field(default_factory=list)
    operating_cf_annual: list[float] = field(default_factory=list)
    capex_annual: list[float] = field(default_factory=list)
    equity_annual: list[float] = field(default_factory=list)
    total_assets_annual: list[float] = field(default_factory=list)
    total_liabilities_annual: list[float] = field(default_factory=list)
    long_term_debt_annual: list[float] = field(default_factory=list)
    eps_diluted_annual: list[float] = field(default_factory=list)
    rd_expense_annual: list[float] = field(default_factory=list)

    # --- Derived annual series ---------------------------------------------
    fcf_annual: list[float] = field(default_factory=list)           # operating_cf − capex
    gross_margin_annual: list[float] = field(default_factory=list)  # %
    operating_margin_annual: list[float] = field(default_factory=list)  # %
    net_margin_annual: list[float] = field(default_factory=list)    # %
    revenue_growth_annual: list[float] = field(default_factory=list)  # YoY %
    eps_growth_annual: list[float] = field(default_factory=list)    # YoY %
    roe_annual: list[float] = field(default_factory=list)           # net_income / equity
    roa_annual: list[float] = field(default_factory=list)           # net_income / total_assets
    debt_to_equity_annual: list[float] = field(default_factory=list)  # LT debt / equity
    shares_outstanding_annual: list[float] = field(default_factory=list)  # absolute shares
    shares_dilution_annual: list[float] = field(default_factory=list)    # YoY % change
    cash_annual: list[float] = field(default_factory=list)               # cash & equivalents
    da_annual: list[float] = field(default_factory=list)                 # depreciation & amortisation
    ebitda_annual: list[float] = field(default_factory=list)             # operating_income + d&a

    # --- Latest-period scalars (from yfinance info) ------------------------
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    pe_trailing: Optional[float] = None
    pe_forward: Optional[float] = None
    beta: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    dividend_yield: Optional[float] = None  # 0.0–1.0 fraction
    avg_volume: Optional[float] = None       # average daily volume (shares)

    # --- Price history (last ≤252 trading days, newest last) ---------------
    close_prices: list[float] = field(default_factory=list)
    spy_close_prices: list[float] = field(default_factory=list)  # SPY closes for relative strength

    # --- Data-quality summary ----------------------------------------------
    years_of_history: int = 0
    data_quality: str = "poor"   # "good" | "partial" | "poor"
    missing_metrics: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _annual_series(df: pd.DataFrame | None) -> tuple[list[int], list[float]]:
    """
    Extract a sorted list of (fiscal_year, value) pairs from a raw EDGAR DataFrame.
    Only 10-K rows; duplicates dropped (most-recently-filed wins, already done upstream).
    Returns (years, values) — both lists, oldest first.
    """
    if df is None or df.empty:
        return [], []
    ann = df[df["form"] == "10-K"].copy()
    if ann.empty:
        return [], []
    ann = ann.sort_values("end")
    years = ann["end"].dt.year.tolist()
    values = ann["val"].tolist()
    return years, values


def _align(
    years_ref: list[int],
    years_src: list[int],
    values_src: list[float],
) -> list[float]:
    """
    Align *values_src* (indexed by *years_src*) to *years_ref*.
    Missing years → NaN.
    """
    src_map = dict(zip(years_src, values_src))
    return [src_map.get(y, float("nan")) for y in years_ref]


def _pct_change(values: list[float]) -> list[float]:
    """Return YoY % change list (first element is always NaN)."""
    result: list[float] = [float("nan")]
    for i in range(1, len(values)):
        prev = values[i - 1]
        curr = values[i]
        if _valid(prev) and prev != 0:
            result.append((curr - prev) / abs(prev) * 100.0)
        else:
            result.append(float("nan"))
    return result


def _ratio(num: list[float], den: list[float]) -> list[float]:
    """Element-wise num/den; result is NaN when denominator is 0 or NaN."""
    out: list[float] = []
    for n, d in zip(num, den):
        if _valid(n) and _valid(d) and d != 0:
            out.append(n / d)
        else:
            out.append(float("nan"))
    return out


def _pct_ratio(num: list[float], den: list[float]) -> list[float]:
    """Element-wise num/den * 100 (for margins)."""
    return [v * 100.0 if _valid(v) else float("nan") for v in _ratio(num, den)]


def _subtract(a: list[float], b: list[float]) -> list[float]:
    """Element-wise a − b; NaN if either is NaN."""
    out: list[float] = []
    for x, y in zip(a, b):
        if _valid(x) and _valid(y):
            out.append(x - y)
        else:
            out.append(float("nan"))
    return out


def _add(a: list[float], b: list[float]) -> list[float]:
    """Element-wise a + b; NaN if either is NaN."""
    out: list[float] = []
    for x, y in zip(a, b):
        if _valid(x) and _valid(y):
            out.append(x + y)
        else:
            out.append(float("nan"))
    return out


def _valid(v: float) -> bool:
    """True iff v is a finite, non-NaN number."""
    try:
        return math.isfinite(v)
    except (TypeError, ValueError):
        return False


def _safe_float(info: dict, key: str) -> Optional[float]:
    v = info.get(key)
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _data_quality(
    years_of_history: int,
    missing: list[str],
    core_metrics: tuple[str, ...] = ("revenue", "net_income", "operating_cf"),
) -> str:
    core_missing = [m for m in missing if m in core_metrics]
    if years_of_history >= 4 and not core_missing:
        return "good"
    if years_of_history >= 2 and len(core_missing) <= 1:
        return "partial"
    return "poor"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalise(
    fundamentals: dict[str, pd.DataFrame],
    price_df: pd.DataFrame,
    info: dict,
    ticker: str = "",
    spy_prices: list[float] | None = None,
) -> NormalisedData:
    """
    Build a NormalisedData object from raw fetched inputs.

    Parameters
    ----------
    fundamentals : dict[str, DataFrame]
        Output of ``fetch_fundamentals()``.
    price_df : DataFrame
        Output of ``fetch_ohlcv()``  (columns: Open/High/Low/Close/Volume).
    info : dict
        Output of ``fetch_info()`` (yfinance .info dict).
    ticker : str
        Ticker symbol — stored for reference only.
    """
    nd = NormalisedData(ticker=ticker.upper())
    missing: list[str] = []

    # ------------------------------------------------------------------
    # 1. Extract annual series for each metric
    # ------------------------------------------------------------------
    raw: dict[str, tuple[list[int], list[float]]] = {}
    for metric in [
        "revenue", "gross_profit", "operating_income", "net_income",
        "operating_cf", "capex", "equity", "total_assets", "total_liabilities",
        "long_term_debt", "eps_diluted", "rd_expense",
        "shares_outstanding", "cash", "da_expense",
    ]:
        df = fundamentals.get(metric)
        years, vals = _annual_series(df)
        raw[metric] = (years, vals)
        if not years:
            missing.append(metric)

    # ------------------------------------------------------------------
    # 2. Determine reference year axis (from revenue, fallback others)
    # ------------------------------------------------------------------
    ref_years: list[int] = []
    for metric in ["revenue", "net_income", "operating_cf", "equity"]:
        years, _ = raw.get(metric, ([], []))
        if years:
            ref_years = years
            break

    if not ref_years:
        nd.missing_metrics = missing
        nd.data_quality = "poor"
        return nd

    nd.years = ref_years

    # ------------------------------------------------------------------
    # 3. Align all series to the reference year axis
    # ------------------------------------------------------------------
    def aligned(metric: str) -> list[float]:
        y, v = raw.get(metric, ([], []))
        return _align(ref_years, y, v)

    rev  = aligned("revenue")
    gp   = aligned("gross_profit")
    oi   = aligned("operating_income")
    ni   = aligned("net_income")
    ocf  = aligned("operating_cf")
    cap  = aligned("capex")
    eq   = aligned("equity")
    ta   = aligned("total_assets")
    tl   = aligned("total_liabilities")
    ltd  = aligned("long_term_debt")
    eps  = aligned("eps_diluted")
    rnd  = aligned("rd_expense")
    shr  = aligned("shares_outstanding")
    csh  = aligned("cash")
    da   = aligned("da_expense")

    # ------------------------------------------------------------------
    # 4. Store raw annual series
    # ------------------------------------------------------------------
    nd.revenue_annual           = rev
    nd.gross_profit_annual      = gp
    nd.operating_income_annual  = oi
    nd.net_income_annual        = ni
    nd.operating_cf_annual      = ocf
    nd.capex_annual             = cap
    nd.equity_annual            = eq
    nd.total_assets_annual      = ta
    nd.total_liabilities_annual = tl
    nd.long_term_debt_annual    = ltd
    nd.eps_diluted_annual       = eps
    nd.rd_expense_annual        = rnd
    nd.shares_outstanding_annual = shr
    nd.cash_annual              = csh
    nd.da_annual                = da

    # ------------------------------------------------------------------
    # 5. Derived series
    # ------------------------------------------------------------------
    nd.fcf_annual              = _subtract(ocf, cap)
    nd.gross_margin_annual     = _pct_ratio(gp,  rev)
    nd.operating_margin_annual = _pct_ratio(oi,  rev)
    nd.net_margin_annual       = _pct_ratio(ni,  rev)
    nd.revenue_growth_annual   = _pct_change(rev)
    nd.eps_growth_annual       = _pct_change(eps)
    nd.roe_annual              = _pct_ratio(ni, eq)
    nd.roa_annual              = _pct_ratio(ni, ta)
    nd.debt_to_equity_annual   = _ratio(ltd, eq)
    nd.shares_dilution_annual  = _pct_change(shr)
    nd.ebitda_annual           = _add(oi, da)

    # ------------------------------------------------------------------
    # 6. Scalar info from yfinance
    # ------------------------------------------------------------------
    nd.current_price  = _safe_float(info, "currentPrice") or _safe_float(info, "regularMarketPrice")
    nd.market_cap     = _safe_float(info, "marketCap")
    nd.pe_trailing    = _safe_float(info, "trailingPE")
    nd.pe_forward     = _safe_float(info, "forwardPE")
    nd.beta           = _safe_float(info, "beta")
    nd.sector         = info.get("sector") or None
    nd.industry       = info.get("industry") or None
    raw_yield         = _safe_float(info, "dividendYield")
    nd.dividend_yield = raw_yield  # already 0.0–1.0 in yfinance
    nd.avg_volume     = _safe_float(info, "averageVolume") or _safe_float(info, "averageDailyVolume10Day")

    # Fallback price from OHLCV if info didn't have it
    if nd.current_price is None and price_df is not None and not price_df.empty:
        nd.current_price = float(price_df["Close"].iloc[-1])

    # Store recent close prices for technical scoring (last 252 trading days)
    if price_df is not None and not price_df.empty:
        closes = price_df["Close"].dropna().tail(252)
        nd.close_prices = [float(v) for v in closes]

    if spy_prices:
        nd.spy_close_prices = list(spy_prices[-252:])

    # ------------------------------------------------------------------
    # 7. Data quality
    # ------------------------------------------------------------------
    nd.years_of_history = len(ref_years)
    nd.missing_metrics  = missing
    nd.data_quality     = _data_quality(nd.years_of_history, missing)

    return nd
