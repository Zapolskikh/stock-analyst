"""
Debug Report — полный трассировочный отчёт по одной акции.

Прогоняет весь пайплайн и записывает в файл reports/<TICKER>_debug.txt
подробный отчёт по каждому шагу: сырые данные → нормализация → классификация
→ скоринг каждого блока → fair value → итог.

Usage:
    python scripts/debug_report.py NVDA
    python scripts/debug_report.py XOM --out my_reports/
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v, fmt=".2f", suffix="", na="n/a"):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return na
    return f"{v:{fmt}}{suffix}"

def _fmt_b(v, na="n/a"):
    """Format large dollar value in billions."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return na
    return f"${v / 1e9:.2f}B"

def _fmt_m(v, na="n/a"):
    """Format large dollar value in millions."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return na
    return f"${v / 1e6:.1f}M"

def _fmt_pct(v, na="n/a"):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return na
    return f"{v:+.1f}%"

def _bar(score_0_10: float, width: int = 10) -> str:
    filled = max(0, min(width, round(score_0_10)))
    return "█" * filled + "░" * (width - filled)

def _series_str(years, values, fmt=".1f", scale=1.0, suffix="") -> str:
    """Render aligned year → value table as a string."""
    if not years:
        return "  (no data)"
    lines = []
    for y, v in zip(years, values):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            lines.append(f"  {y}:  n/a")
        else:
            lines.append(f"  {y}:  {v * scale:{fmt}}{suffix}")
    return "\n".join(lines)

def _section(title: str) -> str:
    return f"\n{'─' * 70}\n  {title}\n{'─' * 70}"

def _header(title: str) -> str:
    return f"\n{'═' * 70}\n  {title}\n{'═' * 70}"


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def build_debug_report(ticker: str) -> str:
    from scripts.load_offline import normalise_offline
    from src.engine.engine import analyse_nd, _HORIZON_WEIGHTS, _check_stop_factors, _rating, _decision
    from src.classifier import classify
    from src.models.benchmarks import get_benchmark
    from src.scoring.quality import score_quality
    from src.scoring.valuation import score_valuation
    from src.scoring.technical import score_technical
    from src.scoring.risk import score_risk
    from src.scoring.style_fit import score_style_fit
    from src.scoring.fair_value import compute_fair_value

    ticker = ticker.upper()
    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines.append(_header(f"DEBUG ANALYSIS REPORT — {ticker}"))
    lines.append(f"  Generated: May 15, 2026")

    # ── Step 1: Load & Normalise ─────────────────────────────────────────────
    lines.append(_section("STEP 1 — NORMALISED DATA"))

    nd = normalise_offline(ticker)

    # Run full pipeline once to get trade_rec and all derived values
    result = analyse_nd(nd)

    lines.append(f"\n  Ticker          : {nd.ticker}")
    lines.append(f"  Sector          : {nd.sector or 'n/a'}")
    lines.append(f"  Industry        : {nd.industry or 'n/a'}")
    lines.append(f"  Years of history: {nd.years_of_history}")
    lines.append(f"  Data quality    : {nd.data_quality}")
    if nd.missing_metrics:
        lines.append(f"  Missing metrics : {', '.join(nd.missing_metrics)}")
    lines.append(f"  TTM as-of       : {nd.ttm_as_of or 'n/a'}")

    lines.append("\n  ── Market / Price ──")
    lines.append(f"  Current price   : {_fmt(nd.current_price, '.2f', '$', '$n/a')}")
    lines.append(f"  Market cap      : {_fmt_b(nd.market_cap)}")
    lines.append(f"  P/E trailing    : {_fmt(nd.pe_trailing, '.1f')}")
    lines.append(f"  P/E forward     : {_fmt(nd.pe_forward, '.1f')}")
    lines.append(f"  Beta            : {_fmt(nd.beta, '.2f')}")
    lines.append(f"  Dividend yield  : {_fmt(nd.dividend_yield * 100 if nd.dividend_yield else None, '.2f', '%')}")
    lines.append(f"  Avg daily vol   : {_fmt(nd.avg_volume, ',.0f', ' shares')}")

    lines.append("\n  ── Analyst Consensus ──")
    lines.append(f"  Forward EPS     : {_fmt(nd.forward_eps, '.2f', '$')}")
    lines.append(f"  Target (median) : {_fmt(nd.analyst_target_median, '.2f', '$')}")
    lines.append(f"  Target (mean)   : {_fmt(nd.analyst_target_mean, '.2f', '$')}")
    lines.append(f"  # analysts      : {nd.analyst_count or 'n/a'}")
    if nd.recommendation_key:
        rec_label = {
            "strong_buy": "Strong Buy", "buy": "Buy", "hold": "Hold",
            "sell": "Sell", "strong_sell": "Strong Sell"
        }.get(nd.recommendation_key, nd.recommendation_key)
        lines.append(f"  Recommendation  : {rec_label}  (mean={_fmt(nd.recommendation_mean, '.2f')})")

    lines.append("\n  ── Market Sentiment ──")
    lines.append(f"  Short ratio     : {_fmt(nd.short_ratio, '.1f', ' days')}  (days to cover)")
    lines.append(f"  Short % float   : {_fmt((nd.short_pct_float or 0) * 100 if nd.short_pct_float else None, '.1f', '%')}")
    lines.append(f"  Institutional   : {_fmt((nd.institutional_ownership or 0) * 100 if nd.institutional_ownership else None, '.1f', '%')}  owned by institutions")
    lines.append(f"  Insider owned   : {_fmt((nd.insider_ownership or 0) * 100 if nd.insider_ownership else None, '.1f', '%')}  owned by insiders")

    lines.append("\n  ── Historical Valuation (P/E from ohlcv) ──")
    if nd.pe_hist_avg:
        lines.append(f"  P/E hist avg    : {_fmt(nd.pe_hist_avg, '.1f')}x")
        lines.append(f"  P/E hist high   : {_fmt(nd.pe_hist_high, '.1f')}x")
        lines.append(f"  P/E hist low    : {_fmt(nd.pe_hist_low, '.1f')}x")
        if nd.pe_trailing and nd.pe_hist_avg:
            rel = (nd.pe_trailing / nd.pe_hist_avg - 1) * 100
            lines.append(f"  vs current P/E  : {nd.pe_trailing:.1f}x  ({rel:+.0f}% vs 5yr avg)")
    else:
        lines.append("  (insufficient ohlcv / EPS overlap to compute)")
    if nd.split_adjusted:
        lines.append(f"  ⚠ Split adj.    : {nd.last_split_factor} on {nd.last_split_date} — shares/EPS normalised")

    if nd.normalized_pe is not None:
        lines.append(f"\n  ── Mid-Cycle Valuation (normalized, 7yr EPS median) ──")
        lines.append(f"  Normalized EPS  : {_fmt(nd.normalized_eps, '.2f', '$')}")
        lines.append(f"  Normalized P/E  : {_fmt(nd.normalized_pe, '.1f')}x  (vs trailing {_fmt(nd.pe_trailing, '.1f')}x)")

    if nd.cashflow_anomaly and nd.cashflow_anomaly_detail:
        lines.append(f"\n  ⚠ CASHFLOW ANOMALY: {nd.cashflow_anomaly_detail}")

    # ── Data Quality Breakdown ──────────────────────────────────────────
    lines.append("\n  ── Data Quality Breakdown ──")
    _DQ_ICONS = {"reliable": "✓", "distorted": "⚠", "limited": "~",
                 "full": "✓", "partial": "~", "minimal": "✗",
                 "native": "✓", "adapted": "~", "unsupported": "✗",
                 "anomaly": "⚠", "sector_n_a": "—"}
    def _dq(label: str, val: str) -> str:
        icon = _DQ_ICONS.get(val, "?")
        return f"  {label:<22}: {icon} {val}"
    lines.append(_dq("Accounting",     nd.dq_accounting))
    lines.append(_dq("Cashflow",       nd.dq_cashflow))
    lines.append(_dq("Valuation",      nd.dq_valuation))
    lines.append(_dq("Historical data",nd.dq_historical))
    lines.append(_dq("Sector fit",     nd.dq_sector_fit))

    lines.append("\n  ── TTM (Trailing Twelve Months) ──")
    lines.append(f"  Revenue         : {_fmt_b(nd.ttm_revenue)}")
    lines.append(f"  Gross profit    : {_fmt_b(nd.ttm_gross_profit)}  →  margin {_fmt(nd.ttm_gross_margin, '.1f', '%')}")
    lines.append(f"  Operating inc.  : {_fmt_b(nd.ttm_operating_income)}  →  margin {_fmt(nd.ttm_operating_margin, '.1f', '%')}")
    lines.append(f"  Net income      : {_fmt_b(nd.ttm_net_income)}  →  margin {_fmt(nd.ttm_net_margin, '.1f', '%')}")
    lines.append(f"  Operating CF    : {_fmt_b(nd.ttm_operating_cf)}")
    lines.append(f"  CapEx           : {_fmt_b(nd.ttm_capex)}")
    lines.append(f"  FCF             : {_fmt_b(nd.ttm_fcf)}  →  margin {_fmt(nd.ttm_fcf_margin, '.1f', '%')}")
    lines.append(f"  EPS diluted     : {_fmt(nd.ttm_eps_diluted, '.2f', '$')}")

    lines.append("\n  ── Annual Revenue ($B) ──")
    lines.append(_series_str(nd.years, nd.revenue_annual, ".2f", 1 / 1e9, "B"))

    lines.append("\n  ── Annual Revenue Growth (YoY %) ──")
    lines.append(_series_str(nd.years, nd.revenue_growth_annual, "+.1f", 1.0, "%"))

    lines.append("\n  ── Annual Gross Margin (%) ──")
    lines.append(_series_str(nd.years, nd.gross_margin_annual, ".1f", 1.0, "%"))

    lines.append("\n  ── Annual Operating Margin (%) ──")
    lines.append(_series_str(nd.years, nd.operating_margin_annual, ".1f", 1.0, "%"))

    lines.append("\n  ── Annual Net Margin (%) ──")
    lines.append(_series_str(nd.years, nd.net_margin_annual, ".1f", 1.0, "%"))

    lines.append("\n  ── Annual Net Income ($B) ──")
    lines.append(_series_str(nd.years, nd.net_income_annual, ".2f", 1 / 1e9, "B"))

    lines.append("\n  ── Annual FCF ($B) ──")
    lines.append(_series_str(nd.years, nd.fcf_annual, ".2f", 1 / 1e9, "B"))

    lines.append("\n  ── Annual EPS diluted ($) ──")
    lines.append(_series_str(nd.years, nd.eps_diluted_annual, ".2f", 1.0, "$"))

    lines.append("\n  ── Annual EPS Growth (YoY %) ──")
    lines.append(_series_str(nd.years, nd.eps_growth_annual, "+.1f", 1.0, "%"))

    lines.append("\n  ── Annual ROE (%) ──")
    lines.append(_series_str(nd.years, nd.roe_annual, ".1f", 1.0, "%"))

    lines.append("\n  ── Annual ROA (%) ──")
    lines.append(_series_str(nd.years, nd.roa_annual, ".1f", 1.0, "%"))

    lines.append("\n  ── Annual Debt-to-Equity (x) ──")
    lines.append(_series_str(nd.years, nd.debt_to_equity_annual, ".2f", 1.0, "x"))

    lines.append("\n  ── Annual Long-term Debt ($B) ──")
    lines.append(_series_str(nd.years, nd.long_term_debt_annual, ".2f", 1 / 1e9, "B"))

    lines.append("\n  ── Annual Equity ($B) ──")
    lines.append(_series_str(nd.years, nd.equity_annual, ".2f", 1 / 1e9, "B"))

    lines.append("\n  ── Annual Total Assets ($B) ──")
    lines.append(_series_str(nd.years, nd.total_assets_annual, ".2f", 1 / 1e9, "B"))

    lines.append("\n  ── Annual Shares Outstanding (M) ──")
    lines.append(_series_str(nd.years, nd.shares_outstanding_annual, ".1f", 1 / 1e6, "M"))

    lines.append("\n  ── Annual Share Dilution YoY (%) ──")
    lines.append(_series_str(nd.years, nd.shares_dilution_annual, "+.1f", 1.0, "%"))

    lines.append("\n  ── Annual EBITDA ($B) ──")
    lines.append(_series_str(nd.years, nd.ebitda_annual, ".2f", 1 / 1e9, "B"))

    lines.append("\n  ── Annual Operating CF ($B) ──")
    lines.append(_series_str(nd.years, nd.operating_cf_annual, ".2f", 1 / 1e9, "B"))

    lines.append("\n  ── Annual CapEx ($B) ──")
    lines.append(_series_str(nd.years, nd.capex_annual, ".2f", 1 / 1e9, "B"))

    lines.append("\n  ── Annual Cash ($B) ──")
    lines.append(_series_str(nd.years, nd.cash_annual, ".2f", 1 / 1e9, "B"))

    lines.append("\n  ── Price History ──")
    cp = nd.close_prices
    if cp:
        lines.append(f"  Bars available  : {len(cp)} trading days")
        lines.append(f"  First close     : ${cp[0]:.2f}")
        lines.append(f"  Last close      : ${cp[-1]:.2f}")
        lines.append(f"  52w high (proxy): ${max(cp):.2f}")
        lines.append(f"  52w low  (proxy): ${min(cp):.2f}")
    else:
        lines.append("  (no price history)")

    # ── Step 2: Classification ────────────────────────────────────────────────
    lines.append(_section("STEP 2 — CLASSIFICATION"))

    cr = classify(nd)
    lines.append(f"\n  Winner          : {cr.company_type.value}")
    lines.append(f"  Confidence      : {cr.confidence * 100:.0f}%")
    lines.append("\n  All type scores:")
    for ctype, score in sorted(cr.scores.items(), key=lambda x: -x[1]):
        bar = _bar(score / 10)
        lines.append(f"    {ctype.value:<25} {score:5.1f}  [{bar}]")
    lines.append("\n  Winning signals:")
    for sig in cr.signals:
        lines.append(f"    • {sig}")

    # ── Step 3: Benchmark ─────────────────────────────────────────────────────
    lines.append(_section("STEP 3 — BENCHMARK"))

    bm = get_benchmark(cr.company_type)
    lines.append(f"\n  Benchmark for   : {cr.company_type.value}")
    lines.append("\n  Block weights (used for overall score):")
    bw = bm.weights
    lines.append(f"    quality        {bw.quality:.2f}")
    lines.append(f"    valuation      {bw.valuation:.2f}")
    lines.append(f"    technical      {bw.technical:.2f}")
    lines.append(f"    risk           {bw.risk:.2f}")
    lines.append(f"    style_fit      {bw.style_fit:.2f}")

    # ── Step 4: Block Scores ──────────────────────────────────────────────────
    lines.append(_section("STEP 4 — BLOCK SCORES"))

    blocks = {
        "quality":   score_quality(nd, bm, cr.company_type),
        "valuation": score_valuation(nd, bm, cr.company_type),
        "technical": score_technical(nd),
        "risk":      score_risk(nd, bm, cr.company_type),
        "style_fit": score_style_fit(nd, bm),
    }

    for name, bs in blocks.items():
        lines.append(f"\n  ── Block: {name.upper()} ──  score={bs.score:.2f}/10  coverage={bs.coverage:.0%}  [{_bar(bs.score)}]")
        if bs.breakdown:
            lines.append("  Sub-metrics:")
            for metric, val in bs.breakdown.items():
                bar = _bar(val)
                lines.append(f"    {metric:<25} {val:5.2f}  [{bar}]")
        if bs.notes:
            lines.append("  Notes:")
            for note in bs.notes:
                lines.append(f"    • {note}")

    # ── Step 5: Horizon Scores ────────────────────────────────────────────────
    lines.append(_section("STEP 5 — HORIZON SCORES"))

    raw = {k: v.score for k, v in blocks.items()}

    lines.append("\n  Horizon weight tables:")
    lines.append(f"  {'block':<12}  {'short':>7}  {'medium':>7}  {'long':>7}")
    lines.append(f"  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*7}")
    for block in ["quality", "valuation", "technical", "risk", "style_fit"]:
        sw = _HORIZON_WEIGHTS["short"][block]
        mw = _HORIZON_WEIGHTS["medium"][block]
        lw = _HORIZON_WEIGHTS["long"][block]
        lines.append(f"  {block:<12}  {sw:>7.2f}  {mw:>7.2f}  {lw:>7.2f}")

    def weighted_avg(scores, weights):
        total_w = sum(weights.values())
        return sum(scores.get(k, 0) * w for k, w in weights.items()) / total_w * 10

    short_score  = weighted_avg(raw, _HORIZON_WEIGHTS["short"])
    medium_score = weighted_avg(raw, _HORIZON_WEIGHTS["medium"])
    long_score   = weighted_avg(raw, _HORIZON_WEIGHTS["long"])

    lines.append(f"\n  Short-term   : {short_score:.1f} / 100")
    lines.append(f"  Medium-term  : {medium_score:.1f} / 100")
    lines.append(f"  Long-term    : {long_score:.1f} / 100")

    lines.append("\n  Overall score (benchmark weights × 10):")
    overall = sum(raw.get(k, 0) * getattr(bw, k) for k in ["quality", "valuation", "technical", "risk", "style_fit"])
    overall_score = overall * 10
    lines.append(f"  = {' + '.join(f'{raw.get(k,0):.2f}×{getattr(bw,k):.2f}' for k in ['quality','valuation','technical','risk','style_fit'])}")
    lines.append(f"  = {overall:.4f}  × 10 = {overall_score:.1f} / 100")

    # ── Step 6: Stop Factors ─────────────────────────────────────────────────
    lines.append(_section("STEP 6 — STOP FACTORS"))

    stop_factors = result.stop_factors
    if stop_factors:
        for sf in stop_factors:
            icon = "🔴" if sf.severity == "critical" else "🟡"
            lines.append(f"\n  {icon} [{sf.severity.upper()}] {sf.name}")
            lines.append(f"     {sf.description}")
    else:
        lines.append("\n  No stop factors triggered.")

    # ── Step 7: Fair Value ────────────────────────────────────────────────────
    lines.append(_section("STEP 7 — FAIR VALUE ESTIMATE"))

    fv = result.fair_value
    if fv is None:
        lines.append("\n  Fair value could not be computed (insufficient data).")
    else:
        lines.append(f"\n  Current price   : ${fv.current_price:.2f}")
        lines.append(f"  Fair value      : ${fv.fair_value:.2f}")
        lines.append(f"  Upside/downside : {fv.upside_str}")
        lines.append(f"  Status          : {fv.status_icon} {fv.status}")
        lines.append("\n  Model breakdown:")
        for model_name, (val, weight) in fv.model_values.items():
            lines.append(f"    {model_name:<15} ${val:>8.2f}  weight={weight:.2%}")
        lines.append("\n  Model assumptions:")
        for a in fv.assumptions:
            lines.append(f"    • {a}")
        if fv.dcf_range is not None:
            bear, base, bull = fv.dcf_range
            b_str = f"${bear:.2f}" if bear is not None else "n/a"
            bull_str = f"${bull:.2f}" if bull is not None else "n/a"
            lines.append(f"\n  DCF Scenario Range:")
            lines.append(f"    Bear (g×0.55, r×1.10)  : {b_str}")
            lines.append(f"    Base                   : ${base:.2f}" if base is not None else "    Base: n/a")
            lines.append(f"    Bull (g×1.45, r×0.92)  : {bull_str}")
            spread = (bull - bear) / base * 100 if (bear and bull and base) else 0
            lines.append(f"    Bear-to-Bull spread    : {spread:.0f}% of base DCF")

    # ── Step 8: Final Result ──────────────────────────────────────────────────
    lines.append(_section("STEP 8 — FINAL RESULT"))

    rating   = result.rating
    decision = result.decision

    lines.append(f"\n  Type            : {cr.company_type.value}  (confidence {cr.confidence * 100:.0f}%)")
    lines.append(f"  Rating             : {rating}")
    lines.append(f"  Overall score      : {overall_score:.1f} / 100")
    lines.append(f"  Investment Quality : {decision}")
    lines.append(f"  Entry Decision     : {result.trade_rec.action if result.trade_rec else 'n/a'}")
    lines.append(f"\n  Short-term      : {short_score:.1f} / 100")
    lines.append(f"  Medium-term     : {medium_score:.1f} / 100")
    lines.append(f"  Long-term       : {long_score:.1f} / 100")

    lines.append("\n  Block scores summary:")
    for name, bs in blocks.items():
        lines.append(f"    {name:<12}  {bs.score:.1f} / 10  [{_bar(bs.score)}]")

    if stop_factors:
        lines.append("\n  Active stop factors:")
        for sf in stop_factors:
            icon = "🔴" if sf.severity == "critical" else "🟡"
            lines.append(f"    {icon} {sf.name}: {sf.description}")

    lines.append("\n  Analyst notes:")
    for name, bs in blocks.items():
        for note in bs.notes:
            lines.append(f"    • [{name}] {note}")

    # ── Step 9: Trade Recommendation ─────────────────────────────────────────
    lines.append(_section("STEP 9 — TRADE RECOMMENDATION"))

    tr = result.trade_rec
    if tr is None:
        lines.append("\n  (not computed)")
    elif tr.action == "Avoid":
        lines.append("\n  🔴 AVOID — не входим в позицию")
        lines.append("\n  Причины:")
        for r in tr.rationale:
            lines.append(f"    ✗ {r}")
    elif tr.action == "Accumulate":
        current = fv.current_price if fv else nd.current_price
        lines.append("\n  🟢 ACCUMULATE — покупка по рынку")
        lines.append(f"\n  Вход           : рыночная цена  ~${current:.2f}" if current else "\n  Вход           : рыночная цена")
        lines.append(f"  Горизонт       : {tr.horizon_label}  (макс. {tr.hold_months} мес.)")
        lines.append(f"  Цель выхода    : ${tr.target_price:.2f}")
        lines.append(f"  Стоп-лосс      : ${tr.stop_price:.2f}")
        if current:
            upside   = (tr.target_price / current - 1) * 100
            downside = (tr.stop_price   / current - 1) * 100
            lines.append(f"  Потенциал      : +{upside:.1f}%  /  риск {downside:.1f}%")
            if downside != 0:
                lines.append(f"  R/R ratio      : {abs(upside / downside):.1f} : 1")
        lines.append("\n  Обоснование:")
        for r in tr.rationale:
            lines.append(f"    • {r}")
    else:  # Accumulate on Pullback
        current = fv.current_price if fv else nd.current_price
        lines.append("\n  🟡 ACCUMULATE ON PULLBACK — ждём лучшей цены")
        if current:
            lines.append(f"\n  Текущая цена   : ${current:.2f}")
        if tr.limit_price is not None:
            lines.append(f"  Лимитная цена  : ${tr.limit_price:.2f}" +
                         (f"  ({(tr.limit_price/current - 1)*100:.1f}% от текущей)" if current else ""))
        if tr.limit_wait_days:
            lines.append(f"  Ждём           : до {tr.limit_wait_days} календарных дней")
        if tr.horizon_label:
            lines.append(f"  Горизонт       : {tr.horizon_label}  (макс. {tr.hold_months} мес.)")
        if tr.target_price is not None:
            lines.append(f"  Цель выхода    : ${tr.target_price:.2f}")
        if tr.stop_price is not None:
            lines.append(f"  Стоп-лосс      : ${tr.stop_price:.2f}")
        entry_px = tr.limit_price or current
        if entry_px and tr.target_price and tr.stop_price:
            upside   = (tr.target_price / entry_px - 1) * 100
            downside = (tr.stop_price   / entry_px - 1) * 100
            lines.append(f"  Потенциал      : +{upside:.1f}%  /  риск {downside:.1f}%  (от лимитной цены)")
            if downside != 0:
                lines.append(f"  R/R ratio      : {abs(upside / downside):.1f} : 1")
        lines.append("\n  Обоснование:")
        for r in tr.rationale:
            lines.append(f"    • {r}")

    # ── Step 10: AI Context Review Prompt ────────────────────────────────────
    lines.append(_section("STEP 10 — AI CONTEXT REVIEW PROMPT"))
    lines.append("\n  Скопируй блок ниже и отправь в AI-модель (без доступа к интернету).")
    lines.append("  AI проверит решение и даст contextual review на основе наших данных.\n")

    import json

    def _sv(v, scale=1.0, digits=2):
        """Safe value: None/NaN → None, else rounded."""
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return None
        return round(float(v) * scale, digits)

    def _recent(lst, n=3):
        """Last n finite values from a list."""
        if not lst:
            return []
        return [_sv(v) for v in lst[-n:] if v is not None and (not isinstance(v, float) or math.isfinite(v))]

    # Build fair-value block

    # Pre-compute moat signal variables (used in payload dict below)
    def _mean_finite(lst, n):
        tail = [v for v in lst[-n:] if v is not None and math.isfinite(v)]
        return sum(tail) / len(tail) if tail else float("nan")

    _gm5       = _mean_finite(nd.gross_margin_annual, 5)
    _gm_recent = _mean_finite(nd.gross_margin_annual, 2)
    _gm_older  = _mean_finite(nd.gross_margin_annual[-5:-2], 2) if len(nd.gross_margin_annual) >= 5 else _gm_recent
    _om5       = _mean_finite(nd.operating_margin_annual, 5)
    _rev5      = _mean_finite(nd.revenue_growth_annual, 5)
    _fcf_conv_vals = [
        ocf / ni if (ni and ni != 0 and math.isfinite(ocf) and math.isfinite(ni)) else float("nan")
        for ocf, ni in zip(nd.operating_cf_annual, nd.net_income_annual)
    ]
    _fcf_conv3 = _mean_finite(_fcf_conv_vals, 3)

    fv_block = None
    if fv:
        fv_block = {
            "current_price":         _sv(fv.current_price),
            "fair_value_composite":  _sv(fv.fair_value),
            "upside_pct":            _sv(fv.discount_pct),
            "status":                fv.status,
            "model_spread_pct":      _sv(fv.model_spread_pct),
            "dcf_range": {
                "bear": _sv(fv.dcf_range[0]) if fv.dcf_range else None,
                "base": _sv(fv.dcf_range[1]) if fv.dcf_range else None,
                "bull": _sv(fv.dcf_range[2]) if fv.dcf_range else None,
            },
            "models": {
                k: {"value": round(v, 2), "weight_pct": round(w * 100, 1)}
                for k, (v, w) in fv.model_values.items()
            },
            "dcf_assumptions": [a for a in fv.assumptions if "DCF" in a or "beta" in a],
            "analyst_assumptions": [a for a in fv.assumptions if "Analyst" in a],
        }

    # Trade rec block
    trade_block = None
    if tr:
        trade_block = {
            "action":       tr.action,
            "horizon":      tr.horizon_label,
            "hold_months":  tr.hold_months,
            "limit_price":  _sv(tr.limit_price) if tr.limit_price else None,
            "target_price": _sv(tr.target_price) if tr.target_price else None,
            "stop_price":   _sv(tr.stop_price) if tr.stop_price else None,
            "rationale":    tr.rationale,
        }

    # Stop factors
    stop_list = [
        {"name": sf.name, "severity": sf.severity, "detail": sf.description}
        for sf in stop_factors
    ]

    # Key analyst notes (all block notes combined)
    analyst_notes_flat = [
        f"[{bname}] {note}"
        for bname, bs in blocks.items()
        for note in bs.notes
    ]

    payload = {
        "ticker": ticker,
        "company_type": cr.company_type.value,
        "classification_confidence_pct": round(cr.confidence * 100),
        "sector":   nd.sector,
        "industry": nd.industry,
        "data_quality": nd.data_quality,

        "scores": {
            "overall":    _sv(overall_score, digits=1),
            "quality":    _sv(blocks["quality"].score,    digits=1),
            "valuation":  _sv(blocks["valuation"].score,  digits=1),
            "technical":  _sv(blocks["technical"].score,  digits=1),
            "risk":       _sv(blocks["risk"].score,       digits=1),
            "style_fit":  _sv(blocks["style_fit"].score,  digits=1),
        },
        "horizon_scores": {
            "short":  _sv(short_score,  digits=1),
            "medium": _sv(medium_score, digits=1),
            "long":   _sv(long_score,   digits=1),
        },
        "quant_decision": decision,

        "market_data": {
            "current_price":          _sv(nd.current_price),
            "market_cap_B":           _sv(nd.market_cap, scale=1/1e9),
            "pe_trailing":            _sv(nd.pe_trailing),
            "pe_forward":             _sv(nd.pe_forward),
            "pe_hist_avg":            _sv(nd.pe_hist_avg),
            "pe_hist_high":           _sv(nd.pe_hist_high),
            "pe_hist_low":            _sv(nd.pe_hist_low),
            "pe_vs_hist_avg_pct":     round((nd.pe_trailing / nd.pe_hist_avg - 1) * 100, 1)
                                      if (nd.pe_trailing and nd.pe_hist_avg) else None,
            "normalized_pe":          _sv(nd.normalized_pe),
            "normalized_eps":         _sv(nd.normalized_eps),
            "cashflow_anomaly":       nd.cashflow_anomaly,
            "cashflow_anomaly_detail": nd.cashflow_anomaly_detail,
            "data_quality_detail": {
                "accounting":   nd.dq_accounting,
                "cashflow":     nd.dq_cashflow,
                "valuation":    nd.dq_valuation,
                "historical":   nd.dq_historical,
                "sector_fit":   nd.dq_sector_fit,
            },
            "beta":                   _sv(nd.beta),
            "dividend_yield_pct":     _sv(nd.dividend_yield, scale=100),
            "analyst_target_median":  _sv(nd.analyst_target_median),
            "analyst_count":          nd.analyst_count,
            "recommendation":         nd.recommendation_key,
            "recommendation_mean":    _sv(nd.recommendation_mean),
            "short_ratio_days":       _sv(nd.short_ratio),
            "short_pct_float":        _sv(nd.short_pct_float, scale=100),
            "institutional_pct":      _sv(nd.institutional_ownership, scale=100),
            "insider_pct":            _sv(nd.insider_ownership, scale=100),
            "split_info":             f"{nd.last_split_factor} on {nd.last_split_date}"
                                      if nd.last_split_factor else None,
        },

        "financials_last_3yr": {
            "revenue_growth_pct":    _recent(nd.revenue_growth_annual),
            "gross_margin_pct":      _recent(nd.gross_margin_annual),
            "operating_margin_pct":  _recent(nd.operating_margin_annual),
            "net_margin_pct":        _recent(nd.net_margin_annual),
            "roe_pct":               _recent(nd.roe_annual),
            "debt_to_equity":        _recent(nd.debt_to_equity_annual),
            "fcf_B":                 _recent([v/1e9 if v and math.isfinite(v) else None
                                              for v in nd.fcf_annual]),
        },

        "ttm": {
            "revenue_B":            _sv(nd.ttm_revenue,          scale=1/1e9),
            "gross_margin_pct":     _sv(nd.ttm_gross_margin),
            "operating_margin_pct": _sv(nd.ttm_operating_margin),
            "net_margin_pct":       _sv(nd.ttm_net_margin),
            "fcf_margin_pct":       _sv(nd.ttm_fcf_margin),
            "fcf_B":                _sv(nd.ttm_fcf,              scale=1/1e9),
            "net_income_B":         _sv(nd.ttm_net_income,       scale=1/1e9),
            "operating_cf_B":       _sv(nd.ttm_operating_cf,     scale=1/1e9),
            "cfo_ni_ratio":         _sv(nd.ttm_operating_cf / nd.ttm_net_income)
                                    if (nd.ttm_net_income and nd.ttm_net_income != 0
                                        and nd.ttm_operating_cf is not None
                                        and math.isfinite(nd.ttm_operating_cf)) else None,
            "accruals_pct_assets":  _sv((nd.ttm_net_income - nd.ttm_operating_cf) /
                                        next((v for v in reversed(nd.total_assets_annual)
                                              if math.isfinite(v)), 1) * 100)
                                    if (nd.ttm_net_income and nd.ttm_operating_cf is not None
                                        and math.isfinite(nd.ttm_operating_cf)) else None,
        },

        "fair_value":          fv_block,
        "stop_factors":        stop_list,
        "trade_recommendation": trade_block,
        "key_quant_notes":     analyst_notes_flat,

        # ── CapEx trend (Пункт 3) ──────────────────────────────────────────
        # Needed for MSFT/GOOGL-type companies with massive data center build-outs:
        # rising CapEx compresses FCF in the short term even when the moat grows.
        "capex_trend": {
            "capex_ttm_B":     _sv(nd.ttm_capex, scale=1/1e9),
            "capex_B_last3yr": _recent([v/1e9 if (v and math.isfinite(v)) else None
                                        for v in nd.capex_annual]),
            "capex_to_revenue_pct_ttm": _sv(nd.ttm_capex / nd.ttm_revenue * 100)
                                        if (nd.ttm_capex and nd.ttm_revenue and nd.ttm_revenue > 0
                                            and math.isfinite(nd.ttm_capex)) else None,
        },

        # ── Moat signals (Пункт 8) ────────────────────────────────────────
        # Quant-observable proxies for economic moat quality.
        # AI layer should interpret these in industry context.
        "moat_signals": {
            "gross_margin_5yr_avg_pct": _sv(_gm5),
            "gross_margin_trend":       "expanding" if _gm_recent > _gm_older else "stable_or_contracting",
            "operating_margin_5yr_avg_pct": _sv(_om5),
            "fcf_conversion_3yr_avg":   _sv(_fcf_conv3),
            "revenue_cagr_5yr_pct":     _sv(_rev5),
        },
    }

    spread_note = (
        f"{fv_block['model_spread_pct']}% spread between valuation models"
        if fv_block else "fair value unavailable"
    )

    prompt = f"""\
You are an institutional equity research analyst performing a CONTEXTUAL REVIEW.
You have NO internet access. Work ONLY with the structured data provided below.

Your role is to complement (not override) a deterministic quantitative scoring model.
The model has already done the math. You provide the CONTEXTUAL LAYER:

1. VALIDATE — Does the quant decision ({decision}) make sense given the data?
2. DISTORTIONS — Are any metrics likely distorted? (one-off events, cyclical peaks,
   accounting artefacts, M&A noise, split contamination, base-effect EPS jumps,
   elevated accruals, CapEx supercycles compressing FCF)
3. NARRATIVE RISKS — Key risks NOT visible in the numbers (sector dynamics,
   product concentration, regulatory exposure, competitive threats by industry type)
4. NARRATIVE TAILWINDS — Structural advantages that may justify a premium or
   sustain above-average growth (platform effects, pricing power, TAM expansion,
   moat signals: gross margin trend, FCF conversion, revenue CAGR)
5. MODEL SPREAD — The fair value models show {spread_note}.
   Comment on WHY they diverge and which anchor is more trustworthy here.
   If dcf_range is available, comment on whether bear/bull range changes the thesis.
6. CONFIDENCE ADJUSTMENT — Integer from -10 to +5 on the overall score.
   DISCIPLINE RULES (strictly enforce):
     • Default is 0. Only deviate if you have SPECIFIC evidence in the data.
     • Negative (-1 to -5): visible risk the quant model underweights (e.g., elevated
       accruals >5%, CapEx spiral reducing FCF, high short interest, sector headwinds).
     • Strongly negative (-6 to -10): only for structural deterioration (NI/OCF split,
       massive insider selling, regulatory crisis, product cliff).
     • Positive (+1 to +3): clear moat expansion visible in margins + FCF conversion.
     • Positive (+4 to +5): rare — exceptional quality AND valuation discount.
     • NEVER positive just because growth is high or the company is well-known.

Respond in EXACTLY this format (no extra sections):
---
VALIDATION: [1-2 sentences]
DISTORTIONS: [bullet list, or "None detected"]
NARRATIVE RISKS: [bullet list]
NARRATIVE TAILWINDS: [bullet list, or "None significant"]
MODEL SPREAD NOTE: [1-2 sentences]
CONFIDENCE ADJUSTMENT: [integer -10..+5] — [one sentence justification citing specific data]
SUMMARY: [2-3 sentences — your overall contextual verdict]
---

DATA:
{json.dumps(payload, indent=2, ensure_ascii=False)}"""

    # Output the prompt indented for the report
    lines.append("=" * 70)
    for pline in prompt.split("\n"):
        lines.append(pline)
    lines.append("=" * 70)

    lines.append(f"\n{'═' * 70}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full debug analysis report for one ticker")
    parser.add_argument("ticker", help="Ticker symbol (must have offline data)")
    parser.add_argument("--out", default="reports", help="Output directory (default: reports/)")
    parser.add_argument("--no-print", dest="no_print", action="store_true",
                        help="Suppress terminal output (only save to file)")
    args = parser.parse_args()

    report = build_debug_report(args.ticker)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.ticker.upper()}_debug.txt"
    out_file.write_text(report, encoding="utf-8")

    if not args.no_print:
        print(report)

    print(f"Report saved → {out_file}")


if __name__ == "__main__":
    main()
