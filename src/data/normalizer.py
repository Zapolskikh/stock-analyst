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

    # --- Analyst consensus (from yfinance) --------------------------------
    forward_eps: Optional[float] = None              # analyst consensus forward EPS
    analyst_target_median: Optional[float] = None    # median analyst price target
    analyst_target_mean: Optional[float] = None      # mean analyst price target
    analyst_count: Optional[int] = None              # number of analyst opinions

    # --- Price history (last ≤252 trading days, newest last) ---------------
    close_prices: list[float] = field(default_factory=list)
    spy_close_prices: list[float] = field(default_factory=list)  # SPY closes for relative strength

    # --- Trailing twelve months (TTM) from quarterly 10-Q data ---------------
    # Flow metrics: sum of last 4 quarterly values (income statement / cash flow)
    ttm_revenue:           Optional[float] = None
    ttm_gross_profit:      Optional[float] = None
    ttm_operating_income:  Optional[float] = None
    ttm_net_income:        Optional[float] = None
    ttm_operating_cf:      Optional[float] = None
    ttm_capex:             Optional[float] = None
    ttm_eps_diluted:       Optional[float] = None
    # Derived TTM
    ttm_fcf:               Optional[float] = None   # ttm_operating_cf − ttm_capex
    ttm_gross_margin:      Optional[float] = None   # %
    ttm_operating_margin:  Optional[float] = None   # %
    ttm_net_margin:        Optional[float] = None   # %
    ttm_fcf_margin:        Optional[float] = None   # %
    ttm_as_of:             Optional[str]   = None   # ISO end-date of latest Q used

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

    # 10-K filings (notably Apple) can include quarterly comparative rows
    # alongside the annual total.  Keep only full-year rows (300–400 days).
    #
    # EXCEPTION: balance sheet (instant) metrics in EDGAR have start == end,
    # giving _days == 0.  When all rows are instant, skip the period filter —
    # applying it would silently drop every balance sheet row (equity, debt …).
    if "start" in ann.columns:
        ann["_days"] = (ann["end"] - ann["start"]).dt.days
        all_instant = (ann["_days"] == 0).all()
        if not all_instant:
            ann = ann[(ann["_days"] >= 300) & (ann["_days"] <= 400)]

    if ann.empty:
        return [], []

    # Deduplicate: keep the most recently filed value for each fiscal year-end
    if "filed" in ann.columns:
        ann = ann.sort_values("filed", ascending=False).drop_duplicates("end", keep="first")

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


def _quarterly_series(df: pd.DataFrame | None) -> tuple[list[str], list[float]]:
    """Extract sorted single-quarter (10-Q) values from a raw EDGAR DataFrame.

    SEC EDGAR 10-Q filings contain both true single-quarter rows (period ≈ 90 days)
    and cumulative YTD rows (period ≈ 180–270 days) under the same form type.
    This function keeps only single-quarter rows by requiring the period length
    to be between 60 and 105 days, eliminating YTD and near-annual periods.

    Returns *(end_dates_iso, values)* — both lists, oldest first.
    """
    if df is None or df.empty:
        return [], []
    q = df[df["form"] == "10-Q"].copy()
    if q.empty:
        return [], []

    # Filter to single-quarter periods only (60–105 days)
    if "start" in q.columns:
        q["_days"] = (q["end"] - q["start"]).dt.days
        q = q[(q["_days"] >= 60) & (q["_days"] <= 105)]

    if q.empty:
        return [], []

    # Deduplicate: keep the most recently filed value for each end date
    if "filed" in q.columns:
        q = q.sort_values("filed", ascending=False).drop_duplicates("end", keep="first")

    q = q.sort_values("end")
    dates = q["end"].dt.strftime("%Y-%m-%d").tolist()
    values = q["val"].tolist()
    return dates, values


def _compute_ttm_flow(values: list[float]) -> Optional[float]:
    """Return the trailing-twelve-month total by summing the last 4 quarters.

    Returns *None* when fewer than 4 valid quarterly values are available.
    """
    recent = [v for v in values[-4:] if _valid(v)]
    if len(recent) < 4:
        return None
    return sum(recent)


def _compute_ttm_aligned(
    num_dates: list[str],
    num_vals: list[float],
    den_dates: list[str],
    den_vals: list[float],
    n: int = 4,
) -> tuple[Optional[float], Optional[float]]:
    """Return (ttm_numerator, ttm_denominator) computed over the same *n* quarters.

    Aligns numerator and denominator by matching end dates so that e.g.
    gross_margin = ttm_gross_profit / ttm_revenue uses exactly the same
    set of quarters. Falls back to None when fewer than *n* common dates exist.
    """
    num_map = dict(zip(num_dates, num_vals))
    den_map = dict(zip(den_dates, den_vals))
    common = sorted(set(num_map) & set(den_map))
    if len(common) < n:
        return None, None
    recent = common[-n:]
    num_sum = sum(num_map[d] for d in recent if _valid(num_map[d]))
    den_sum = sum(den_map[d] for d in recent if _valid(den_map[d]))
    if not _valid(num_sum) or not _valid(den_sum):
        return None, None
    return num_sum, den_sum


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
        "revenue", "gross_profit", "cost_of_revenue", "operating_income", "net_income",
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
    # gross_profit: prefer direct tag; fall back to revenue - cost_of_revenue
    _gp_direct = aligned("gross_profit")
    _cogs      = aligned("cost_of_revenue")
    gp = [
        gv if _valid(gv) else (rv - cv if _valid(rv) and _valid(cv) else float("nan"))
        for gv, rv, cv in zip(_gp_direct, rev, _cogs)
    ]
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
    raw_yield = _safe_float(info, "dividendYield")
    # yfinance sometimes returns dividend yield as a percentage (e.g. 2.0 for 2%)
    # rather than as a decimal fraction (0.02).  Values > 0.25 are impossible as
    # fractions for any normal equity, so divide by 100 to normalise to 0–1.
    if raw_yield is not None and raw_yield > 0.25:
        raw_yield = raw_yield / 100.0
    nd.dividend_yield = raw_yield
    nd.avg_volume = (
        _safe_float(info, "averageVolume")
        or _safe_float(info, "averageDailyVolume10Day")
    )
    nd.forward_eps             = _safe_float(info, "forwardEps")
    nd.analyst_target_median   = _safe_float(info, "targetMedianPrice")
    nd.analyst_target_mean     = _safe_float(info, "targetMeanPrice")
    analyst_count = _safe_float(info, "numberOfAnalystOpinions")
    nd.analyst_count = int(analyst_count) if analyst_count is not None else None

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

    # ------------------------------------------------------------------
    # 8. TTM from quarterly (10-Q) filings
    # ------------------------------------------------------------------
    _ttm_flow_metrics = [
        ("revenue",          "ttm_revenue"),
        ("gross_profit",     "ttm_gross_profit"),
        ("operating_income", "ttm_operating_income"),
        ("net_income",       "ttm_net_income"),
        ("operating_cf",     "ttm_operating_cf"),
        ("capex",            "ttm_capex"),
        ("eps_diluted",      "ttm_eps_diluted"),
    ]
    for metric, attr in _ttm_flow_metrics:
        _, qvals = _quarterly_series(fundamentals.get(metric))
        setattr(nd, attr, _compute_ttm_flow(qvals))

    # Derived TTM values
    if nd.ttm_operating_cf is not None and nd.ttm_capex is not None:
        nd.ttm_fcf = nd.ttm_operating_cf - nd.ttm_capex

    # TTM margins: use aligned TTM (same quarters for numerator and denominator)
    # to avoid mixing misaligned quarters when revenue and numerator series have
    # different end-dates (common in SEC EDGAR XBRL data across companies).
    rev_dates, rev_vals = _quarterly_series(fundamentals.get("revenue"))

    def _ttm_margin_aligned(numerator_metric: str) -> Optional[float]:
        num_dates, num_vals = _quarterly_series(fundamentals.get(numerator_metric))
        num_ttm, den_ttm = _compute_ttm_aligned(num_dates, num_vals, rev_dates, rev_vals)
        if num_ttm is None or den_ttm is None or den_ttm == 0:
            return None
        return num_ttm / den_ttm * 100.0

    nd.ttm_gross_margin    = _ttm_margin_aligned("gross_profit")
    nd.ttm_operating_margin = _ttm_margin_aligned("operating_income")
    nd.ttm_net_margin      = _ttm_margin_aligned("net_income")
    if nd.ttm_fcf is not None:
        # FCF margin: use the same revenue TTM that was independently computed
        rev_ttm = nd.ttm_revenue
        if rev_ttm and rev_ttm > 0:
            nd.ttm_fcf_margin = nd.ttm_fcf / rev_ttm * 100.0

    # Record the end-date of the latest quarterly filing used for TTM
    if rev_dates:
        nd.ttm_as_of = rev_dates[-1]

    return nd
