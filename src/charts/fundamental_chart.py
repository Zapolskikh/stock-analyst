"""
Fundamental analysis charts built from SEC EDGAR data.

Public entry point:

    build_fundamental_charts(fundamentals, ticker, out_dir)
        → list[tuple[chart_name, saved_html_path]]

Generates up to 6 charts depending on available data:
    1. Revenue & YoY Growth
    2. Profitability  (Revenue / Gross Profit / Operating Income / Net Income)
    3. Margins %      (Gross / Operating / Net)
    4. Cash Flow      (Operating CF / CapEx / Free Cash Flow)
    5. Balance Sheet  (Equity stack + Long-term Debt overlay)
    6. EPS            (Diluted / Basic)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _annual(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Return deduplicated annual (10-K) rows sorted by period-end date.
    Keeps the most recently filed value when multiple accessions cover
    the same fiscal year-end.
    """
    if df is None or df.empty:
        return None
    ann = df[df["form"] == "10-K"].copy()
    if ann.empty:
        return None
    if "filed" in ann.columns:
        ann = ann.sort_values("filed", ascending=False)
    ann = ann.drop_duplicates("end", keep="first").sort_values("end")
    return ann.reset_index(drop=True)


def _B(series: pd.Series) -> pd.Series:
    """Convert USD → billions (float)."""
    return series / 1e9


_TMPL = "plotly_white"
_HOVER = "x unified"


# ---------------------------------------------------------------------------
# Individual chart builders
# ---------------------------------------------------------------------------

def _revenue_chart(fundamentals: dict, ticker: str) -> go.Figure:
    df = _annual(fundamentals.get("revenue"))
    if df is None:
        return go.Figure()

    df = df.copy()
    df["yoy"] = df["val"].pct_change() * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=df["end"].dt.year, y=_B(df["val"]),
            name="Revenue (B$)", marker_color="#2196F3",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["end"].dt.year, y=df["yoy"],
            name="YoY Growth %",
            mode="lines+markers",
            line=dict(color="#FF9800", width=2),
            marker=dict(size=7),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title=f"{ticker} — Annual Revenue & YoY Growth",
        template=_TMPL, hovermode=_HOVER,
    )
    fig.update_yaxes(title_text="Revenue (B$)",  secondary_y=False)
    fig.update_yaxes(title_text="YoY Growth %",  secondary_y=True)
    return fig


def _profitability_chart(fundamentals: dict, ticker: str) -> go.Figure:
    configs = [
        ("revenue",          "Revenue",          "#2196F3"),
        ("gross_profit",     "Gross Profit",     "#4CAF50"),
        ("operating_income", "Operating Income", "#FF9800"),
        ("net_income",       "Net Income",       "#9C27B0"),
    ]
    fig = go.Figure()
    for metric, label, color in configs:
        df = _annual(fundamentals.get(metric))
        if df is None:
            continue
        fig.add_trace(go.Bar(
            x=df["end"].dt.year, y=_B(df["val"]),
            name=label, marker_color=color,
        ))
    fig.update_layout(
        title=f"{ticker} — Profitability (Annual, B$)",
        barmode="group", template=_TMPL, hovermode=_HOVER,
        yaxis_title="USD (B)",
    )
    return fig


def _margins_chart(fundamentals: dict, ticker: str) -> go.Figure:
    rev_df = _annual(fundamentals.get("revenue"))
    if rev_df is None:
        return go.Figure()

    rev = rev_df.set_index("end")["val"]
    configs = [
        ("gross_profit",     "Gross Margin %",     "#4CAF50"),
        ("operating_income", "Operating Margin %", "#FF9800"),
        ("net_income",       "Net Margin %",       "#9C27B0"),
    ]
    fig = go.Figure()
    for metric, label, color in configs:
        df = _annual(fundamentals.get(metric))
        if df is None:
            continue
        margin = (df.set_index("end")["val"] / rev * 100).dropna()
        if margin.empty:
            continue
        fig.add_trace(go.Scatter(
            x=[d.year for d in margin.index], y=margin.values,
            name=label,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=7),
        ))
    fig.update_layout(
        title=f"{ticker} — Profit Margins % (Annual)",
        template=_TMPL, hovermode=_HOVER,
        yaxis_title="Margin %",
    )
    return fig


def _cashflow_chart(fundamentals: dict, ticker: str) -> go.Figure:
    ocf_df   = _annual(fundamentals.get("operating_cf"))
    capex_df = _annual(fundamentals.get("capex"))

    fig = go.Figure()

    if ocf_df is not None:
        fig.add_trace(go.Bar(
            x=ocf_df["end"].dt.year, y=_B(ocf_df["val"]),
            name="Operating CF", marker_color="#2196F3",
        ))

    if capex_df is not None:
        # XBRL stores CapEx as a positive payment amount → show as outflow
        fig.add_trace(go.Bar(
            x=capex_df["end"].dt.year, y=-_B(capex_df["val"]),
            name="CapEx", marker_color="#ef5350",
        ))

    if ocf_df is not None and capex_df is not None:
        ocf   = ocf_df.set_index("end")["val"]
        capex = capex_df.set_index("end")["val"]
        fcf   = (ocf - capex).dropna()
        if not fcf.empty:
            fig.add_trace(go.Scatter(
                x=[d.year for d in fcf.index], y=_B(fcf),
                name="Free Cash Flow",
                mode="lines+markers",
                line=dict(color="#FF9800", width=2.5),
                marker=dict(size=8),
            ))

    fig.update_layout(
        title=f"{ticker} — Cash Flow (Annual, B$)",
        barmode="relative", template=_TMPL, hovermode=_HOVER,
        yaxis_title="USD (B)",
    )
    return fig


def _balance_sheet_chart(fundamentals: dict, ticker: str) -> go.Figure:
    fig = go.Figure()

    for metric, label, color in [
        ("equity",           "Equity",      "#4CAF50"),
        ("total_liabilities","Liabilities", "#ef5350"),
    ]:
        df = _annual(fundamentals.get(metric))
        if df is None:
            continue
        fig.add_trace(go.Bar(
            x=df["end"].dt.year, y=_B(df["val"]),
            name=label, marker_color=color,
        ))

    debt_df = _annual(fundamentals.get("long_term_debt"))
    if debt_df is not None:
        fig.add_trace(go.Scatter(
            x=debt_df["end"].dt.year, y=_B(debt_df["val"]),
            name="Long-Term Debt",
            mode="lines+markers",
            line=dict(color="#FF9800", width=2),
            marker=dict(size=7),
        ))

    fig.update_layout(
        title=f"{ticker} — Balance Sheet (Annual, B$)",
        barmode="stack", template=_TMPL, hovermode=_HOVER,
        yaxis_title="USD (B)",
    )
    return fig


def _eps_chart(fundamentals: dict, ticker: str) -> go.Figure:
    fig = go.Figure()
    for metric, label, color in [
        ("eps_diluted", "EPS Diluted", "#2196F3"),
        ("eps_basic",   "EPS Basic",   "#4CAF50"),
    ]:
        df = _annual(fundamentals.get(metric))
        if df is None:
            continue
        fig.add_trace(go.Bar(
            x=df["end"].dt.year, y=df["val"],
            name=label, marker_color=color,
        ))
    fig.update_layout(
        title=f"{ticker} — EPS (Annual, USD/share)",
        barmode="group", template=_TMPL, hovermode=_HOVER,
        yaxis_title="EPS (USD)",
    )
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_fundamental_charts(
    fundamentals: dict[str, pd.DataFrame],
    ticker: str,
    out_dir: Path,
) -> list[tuple[str, Path]]:
    """
    Build and save all available fundamental charts as interactive HTML files.

    Returns a list of (chart_name, saved_path) for each chart produced.
    Charts with no data are silently skipped.
    """
    F = fundamentals  # shorthand

    tasks: list[tuple[str, bool, object, str]] = [
        (
            "Revenue & Growth",
            "revenue" in F,
            lambda: _revenue_chart(F, ticker),
            "revenue.html",
        ),
        (
            "Profitability",
            any(k in F for k in ("revenue", "gross_profit", "net_income")),
            lambda: _profitability_chart(F, ticker),
            "profitability.html",
        ),
        (
            "Margins %",
            "revenue" in F and any(k in F for k in ("gross_profit", "net_income")),
            lambda: _margins_chart(F, ticker),
            "margins.html",
        ),
        (
            "Cash Flow",
            any(k in F for k in ("operating_cf", "capex")),
            lambda: _cashflow_chart(F, ticker),
            "cashflow.html",
        ),
        (
            "Balance Sheet",
            any(k in F for k in ("total_liabilities", "equity")),
            lambda: _balance_sheet_chart(F, ticker),
            "balance_sheet.html",
        ),
        (
            "EPS",
            any(k in F for k in ("eps_diluted", "eps_basic")),
            lambda: _eps_chart(F, ticker),
            "eps.html",
        ),
    ]

    results: list[tuple[str, Path]] = []
    for name, condition, builder, filename in tasks:
        if not condition:
            continue
        fig: go.Figure = builder()
        if not fig.data:
            continue
        path = out_dir / filename
        fig.write_html(str(path))
        results.append((name, path))

    return results
