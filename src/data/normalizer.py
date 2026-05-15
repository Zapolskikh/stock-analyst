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

    # --- Market sentiment (from yfinance) ----------------------------------
    short_ratio: Optional[float] = None              # days to cover (short interest / avg volume)
    short_pct_float: Optional[float] = None          # short interest as % of float (0.0–1.0)
    institutional_ownership: Optional[float] = None  # % held by institutions (0.0–1.0)
    insider_ownership: Optional[float] = None        # % held by insiders (0.0–1.0)
    recommendation_key: Optional[str] = None         # "strong_buy" | "buy" | "hold" | "sell" | "strong_sell"
    recommendation_mean: Optional[float] = None      # 1.0 (Strong Buy) → 5.0 (Strong Sell)

    # --- Split info (from yfinance) ----------------------------------------
    last_split_factor: Optional[str] = None          # e.g. "10:1"
    last_split_date: Optional[str] = None            # ISO date string, e.g. "2024-06-10"
    split_adjusted: bool = False                     # True if shares/EPS series were adjusted

    # --- Historical valuation (computed from ohlcv + annual EPS) ----------
    pe_hist_avg: Optional[float] = None              # average P/E over available ohlcv history
    pe_hist_high: Optional[float] = None             # peak P/E over available ohlcv history
    pe_hist_low: Optional[float] = None              # trough P/E over available ohlcv history

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

    # --- Cyclical / mid-cycle normalization --------------------------------
    # For cyclical companies (energy, autos, materials) current-year earnings
    # can be 2-5× higher or lower than mid-cycle norm due to commodity prices.
    # normalized_eps = median of last 7yr EPS (smooths peaks and troughs).
    # normalized_pe  = current_price / normalized_eps.
    normalized_eps: Optional[float] = None
    normalized_pe:  Optional[float] = None

    # --- Cashflow anomaly flag ---------------------------------------------
    # Set when TTM OCF deviates significantly from 5yr historical average.
    # Used by stop factors and AI payload to flag one-off distortions.
    cashflow_anomaly: bool = False
    cashflow_anomaly_detail: Optional[str] = None

    # --- Data-quality summary ----------------------------------------------
    years_of_history: int = 0
    data_quality: str = "poor"   # "good" | "partial" | "poor"
    missing_metrics: list[str] = field(default_factory=list)

    # --- Granular data-quality dimensions ----------------------------------
    # Each dimension is independently flagged so the AI and engine can act on
    # specific weaknesses rather than a single coarse "dq" label.
    dq_accounting: str = "reliable"    # "reliable" | "distorted" | "limited"
    dq_cashflow:   str = "reliable"    # "reliable" | "anomaly"   | "sector_n_a"
    dq_valuation:  str = "reliable"    # "reliable" | "distorted" | "limited"
    dq_historical: str = "full"        # "full" | "partial" | "minimal"
    dq_sector_fit: str = "native"      # "native" | "adapted" | "unsupported"


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
    splits_df: pd.DataFrame | None = None,
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

    # Market sentiment
    nd.short_ratio             = _safe_float(info, "shortRatio")
    nd.short_pct_float         = _safe_float(info, "shortPercentOfFloat")
    nd.institutional_ownership = _safe_float(info, "heldPercentInstitutions")
    nd.insider_ownership       = _safe_float(info, "heldPercentInsiders")
    nd.recommendation_key      = info.get("recommendationKey") or None
    nd.recommendation_mean     = _safe_float(info, "recommendationMean")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Split normalization: full backward-adjustment pipeline
    # ------------------------------------------------------------------
    # SEC EDGAR stores raw (non-split-adjusted) values. To make the entire
    # historical series comparable in today's share terms we must apply ALL
    # splits in reverse-chronological order.
    #
    # Algorithm:
    #   1. Collect all known splits sorted newest → oldest.
    #   2. For each split at date D with ratio R:
    #      - Compute cumulative factor for years BEFORE D.
    #      - Divide shares by R (pre-split shares were fewer → scale down to
    #        today's per-share equivalent).
    #      - Multiply EPS by R (pre-split EPS was higher → scale up to today's
    #        per-share equivalent after dilution).
    #   3. Re-derive dependent series (dilution %, EPS growth) from adjusted data.
    #
    # Sources of split data (in priority order):
    #   A. splits_df — full yfinance history saved by fetch_offline_data.py
    #   B. info["lastSplitFactor"] + info["lastSplitDate"] — legacy fallback
    #      (only the most recent split, kept for backwards compatibility when
    #       yf_splits.parquet has not been fetched yet).

    # Build splits list: list of (split_year: int, ratio: float), newest first
    _splits: list[tuple[int, float]] = []

    if splits_df is not None and not splits_df.empty and "ratio" in splits_df.columns:
        # Primary source: full history from yf_splits.parquet
        for _, _row in splits_df.sort_values("date", ascending=False).iterrows():
            try:
                _dt  = pd.Timestamp(_row["date"])
                _rat = float(_row["ratio"])
                if _rat > 1.0:   # ignore reverse-splits for now (< 1.0)
                    _splits.append((_dt.year, _rat))
            except Exception:
                continue
        # Store metadata from the most recent split
        if _splits:
            _most_recent_dt = splits_df.sort_values("date").iloc[-1]
            nd.last_split_factor = f"{int(_most_recent_dt['ratio'])}:1"
            nd.last_split_date   = pd.Timestamp(_most_recent_dt["date"]).strftime("%Y-%m-%d")
    else:
        # Legacy fallback: single split from info.json
        _raw_sf = info.get("lastSplitFactor")
        _raw_sd = info.get("lastSplitDate")
        if _raw_sf and _raw_sd:
            try:
                _parts       = str(_raw_sf).split(":")
                _split_ratio = float(_parts[0]) / float(_parts[1])
                _sdt         = pd.Timestamp(_raw_sd, unit="s")
                if _split_ratio > 1.0:
                    _splits.append((_sdt.year, _split_ratio))
                nd.last_split_factor = _raw_sf
                nd.last_split_date   = _sdt.strftime("%Y-%m-%d")
            except Exception:
                pass

    if _splits:
        # Apply each split cumulatively from newest to oldest.
        # Goal: express ALL historical values in post-all-splits per-share terms.
        #
        # For SHARES OUTSTANDING (total count):
        #   A split multiplies the share count.  Pre-split years had FEWER shares.
        #   To express them in post-split terms: multiply by the split factor.
        #   Example: 612M shares (pre-4:1) × 4 × 10 = 24,480M post-all-splits.
        #
        # For EPS (earnings per share):
        #   A split halves EPS (same earnings, more shares).  Pre-split EPS was HIGHER.
        #   To express in post-split per-share terms: divide by the split factor.
        #   Example: $1.13/share (pre-4:1) / 4 / 10 = $0.028 post-all-splits per-share.
        #
        # EDGAR RETROACTIVE ADJUSTMENT PROBLEM:
        #   SEC EDGAR sometimes retroactively updates prior-year data to reflect a split
        #   (especially in "comparative" columns of subsequent 10-K filings).
        #   This means the split JUMP may appear at year Y-1 or Y-2 instead of year Y.
        #   We detect the actual jump position by scanning ±3 years around the split date
        #   and only adjust years BEFORE the detected jump (not years already adjusted).

        def _detect_actual_jump_year(
            shares: list[float], yrs: list[int], nominal_split_yr: int, ratio: float
        ) -> int:
            """
            Find the year in the data where the split jump actually appears.
            Returns the nominal split year as fallback if jump is not detected.
            """
            for offset in range(-3, 4):
                yr_candidate = nominal_split_yr + offset
                if yr_candidate not in yrs:
                    continue
                idx = yrs.index(yr_candidate)
                if idx == 0:
                    continue
                prev_v = shares[idx - 1]
                curr_v = shares[idx]
                if not (_valid(prev_v) and _valid(curr_v) and prev_v > 0):
                    continue
                actual_ratio = curr_v / prev_v
                # Match if within 35% of expected ratio
                if abs(actual_ratio - ratio) / ratio <= 0.35:
                    return yr_candidate
            return nominal_split_yr   # fallback: no clear jump found

        _adjusted = False
        # Work on a mutable copy of years list for index lookups
        _yr_list = list(nd.years)

        for _sp_year, _sp_factor in _splits:
            _actual_jump_yr = _detect_actual_jump_year(
                nd.shares_outstanding_annual, _yr_list, _sp_year, _sp_factor
            )
            # Adjust only years BEFORE the detected jump year (those are genuinely pre-split)
            for _i, _yr in enumerate(_yr_list):
                if _yr < _actual_jump_yr:
                    if _valid(nd.shares_outstanding_annual[_i]):
                        nd.shares_outstanding_annual[_i] *= _sp_factor
                        _adjusted = True
                    if _valid(nd.eps_diluted_annual[_i]):
                        nd.eps_diluted_annual[_i] /= _sp_factor

        if _adjusted:
            nd.shares_dilution_annual = _pct_change(nd.shares_outstanding_annual)
            nd.eps_growth_annual      = _pct_change(nd.eps_diluted_annual)
            nd.split_adjusted         = True

    # ------------------------------------------------------------------
    # Historical P/E from ohlcv + annual EPS
    # ------------------------------------------------------------------
    # For each fiscal year where we have both EPS (10-K) and a price close
    # near the filing date, compute trailing P/E.  We look up the stock price
    # within ±45 days of the 10-K end date using the ohlcv DataFrame.
    # Requires split-adjusted EPS for meaningful comparison.
    if price_df is not None and not price_df.empty and nd.eps_diluted_annual:
        _hist_pe_vals: list[float] = []
        # Need the raw annual EPS dates to look up the right price
        _eps_df = fundamentals.get("eps_diluted")
        if _eps_df is not None and not _eps_df.empty:
            _eps_ann = _eps_df[_eps_df["form"] == "10-K"].copy()
            if "start" in _eps_ann.columns:
                _eps_ann["_days"] = (_eps_ann["end"] - _eps_ann["start"]).dt.days
                _all_instant = (_eps_ann["_days"] == 0).all()
                if not _all_instant:
                    _eps_ann = _eps_ann[(_eps_ann["_days"] >= 300) & (_eps_ann["_days"] <= 400)]
            if "filed" in _eps_ann.columns:
                _eps_ann = _eps_ann.sort_values("filed", ascending=False).drop_duplicates("end", keep="first")
            _eps_ann = _eps_ann.sort_values("end")

            # Build a lookup: fiscal year → price at filing date
            _price_idx = pd.DatetimeIndex(price_df.index)
            for _, _row in _eps_ann.iterrows():
                _fy_end  = pd.Timestamp(_row["end"])
                _fy_eps  = float(_row["val"])
                if not _valid(_fy_eps) or _fy_eps <= 0:
                    continue
                # Apply cumulative split adjustment to raw EPS for historical P/E.
                # For each split that happened AFTER this fiscal year end,
                # the EPS was in pre-split (higher) per-share terms → divide to get
                # post-split equivalent (lower EPS per more shares).
                if nd.split_adjusted and _splits:
                    for _sp_yr, _sp_f in _splits:
                        if _fy_end.year < _sp_yr:
                            _fy_eps /= _sp_f

                # Find closest available price within ±45 days of fiscal year end
                _lo = _fy_end - pd.Timedelta(days=45)
                _hi = _fy_end + pd.Timedelta(days=45)
                _mask = (_price_idx >= _lo) & (_price_idx <= _hi)
                _window = price_df.loc[_mask]
                if _window.empty:
                    continue
                # Use the price closest to the fiscal year end date
                _closest_idx = ((_price_idx[_mask] - _fy_end).asi8.astype("int64").__abs__().argmin())
                _fy_price = float(_window["Close"].iloc[_closest_idx])
                if _fy_price > 0:
                    _pe = _fy_price / _fy_eps
                    if 3.0 < _pe < 300.0:   # sanity range
                        _hist_pe_vals.append(_pe)

        if _hist_pe_vals:
            nd.pe_hist_avg  = round(sum(_hist_pe_vals) / len(_hist_pe_vals), 1)
            nd.pe_hist_high = round(max(_hist_pe_vals), 1)
            nd.pe_hist_low  = round(min(_hist_pe_vals), 1)
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

    # ------------------------------------------------------------------
    # 9. Mid-cycle normalized EPS + PE (for cyclical sectors)
    # ------------------------------------------------------------------
    # Uses median of last 7 years of EPS to smooth commodity / economic cycles.
    # Median is more robust than mean: it ignores both boom-year peaks (XOM 2022)
    # and recession troughs (XOM 2020 -$5.25). At least 3 years required.
    _eps_hist = [v for v in nd.eps_diluted_annual if math.isfinite(v) and v != 0]
    if len(_eps_hist) >= 3:
        _eps7 = sorted(_eps_hist[-7:])
        nd.normalized_eps = _eps7[len(_eps7) // 2]    # median
        if nd.current_price and math.isfinite(nd.current_price) and nd.normalized_eps > 0:
            nd.normalized_pe = round(nd.current_price / nd.normalized_eps, 1)

    # ------------------------------------------------------------------
    # 10. Cashflow anomaly detection
    # ------------------------------------------------------------------
    # Flags when TTM OCF deviates sharply from 5yr historical average.
    # This catches one-off distortions that would mislead DCF and quality scoring:
    #   KO TTM OCF = −$2.5B vs 5yr avg $9.8B  →  anomaly (working capital spike)
    #   XOM TTM OCF = $52B vs 5yr avg $47B     →  no anomaly
    # Threshold: |TTM - hist_mean| > 2 × hist_std AND |deviation_pct| > 40%
    if nd.ttm_operating_cf is not None and math.isfinite(nd.ttm_operating_cf):
        _ocf_hist = [v for v in nd.operating_cf_annual[-5:] if math.isfinite(v)]
        if len(_ocf_hist) >= 3:
            import statistics as _stats
            _ocf_mean = _stats.mean(_ocf_hist)
            _ocf_std  = _stats.stdev(_ocf_hist) if len(_ocf_hist) >= 2 else 0.0
            _ocf_dev  = nd.ttm_operating_cf - _ocf_mean
            _dev_pct  = abs(_ocf_dev / _ocf_mean) * 100 if _ocf_mean != 0 else 0.0
            if (_ocf_std > 0 and abs(_ocf_dev) > 2 * _ocf_std and _dev_pct > 40):
                nd.cashflow_anomaly = True
                direction = "surge" if _ocf_dev > 0 else "collapse"
                nd.cashflow_anomaly_detail = (
                    f"TTM OCF ${nd.ttm_operating_cf/1e9:.1f}B vs 5yr avg "
                    f"${_ocf_mean/1e9:.1f}B (±${_ocf_std/1e9:.1f}B) — "
                    f"{_dev_pct:.0f}% {direction}; may distort DCF and quality metrics"
                )

    # ── Section 11: Granular data quality ────────────────────────────────
    # dq_accounting / dq_valuation: PE distortion check
    _pe_t = nd.pe_trailing
    _pe_f = nd.pe_forward
    if (
        _pe_t is not None and _pe_f is not None
        and math.isfinite(_pe_t) and math.isfinite(_pe_f)
        and _pe_f > 0 and _pe_t > 3 * _pe_f and _pe_f < 25
    ):
        nd.dq_accounting = "distorted"
        nd.dq_valuation  = "distorted"

    # dq_cashflow: anomaly flag (set in section 10 above)
    if nd.cashflow_anomaly:
        nd.dq_cashflow = "anomaly"

    # dq_historical: based on years of history
    if nd.years_of_history >= 7:
        nd.dq_historical = "full"
    elif nd.years_of_history >= 3:
        nd.dq_historical = "partial"
    else:
        nd.dq_historical = "minimal"

    # dq_sector_fit: defaulted to "native" here; overridden in engine.py
    # after company_type classification:
    #   FINANCIAL       → "unsupported"  (GAAP metrics ill-suited)
    #   CYCLICAL/PHARMA → "adapted"      (metrics applied with modifications)
    #   others          → "native"

    return nd
