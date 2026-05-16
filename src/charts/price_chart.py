"""
Technical price chart — Candlestick + MA + Volume + RSI + MACD.

build_price_chart(df, ticker) → plotly Figure (4-panel layout):
  Row 1 (50 %): Candlestick with MA20 / MA50 / MA200
  Row 2 (15 %): Volume bars (green/red)
  Row 3 (17 %): RSI(14) with overbought/oversold bands
  Row 4 (18 %): MACD(12,26,9) histogram + signal line
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Technical indicator calculations
# ---------------------------------------------------------------------------

def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _calc_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


# ---------------------------------------------------------------------------
# Chart builder
# ---------------------------------------------------------------------------

def build_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Build an interactive 4-panel price chart for *ticker*.

    *df* must have columns: Open, High, Low, Close, Volume
    with a timezone-naive DatetimeIndex.
    """
    close = df["Close"]
    n = len(df)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.50, 0.15, 0.175, 0.175],
        subplot_titles=[
            f"{ticker}  —  Price & Moving Averages",
            "Volume",
            "RSI (14)",
            "MACD (12 / 26 / 9)",
        ],
    )

    # --- Row 1: Candlestick -------------------------------------------------
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"],   close=close,
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
        ),
        row=1, col=1,
    )

    for period, color in [(20, "#2196F3"), (50, "#FF9800"), (200, "#E91E63")]:
        if n >= period:
            ma = close.rolling(period).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=ma, name=f"MA{period}",
                    mode="lines",
                    line=dict(color=color, width=1.3),
                ),
                row=1, col=1,
            )

    # --- Row 2: Volume -------------------------------------------------------
    vol_colors = [
        "#ef5350" if c < o else "#26a69a"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["Volume"],
            name="Volume", marker_color=vol_colors, showlegend=False,
        ),
        row=2, col=1,
    )

    # --- Row 3: RSI ----------------------------------------------------------
    rsi = _calc_rsi(close)
    fig.add_trace(
        go.Scatter(
            x=df.index, y=rsi, name="RSI",
            mode="lines", line=dict(color="#9C27B0", width=1.3),
        ),
        row=3, col=1,
    )
    # Overbought / oversold bands
    for level, color in [(70, "rgba(239,83,80,0.25)"), (30, "rgba(38,166,154,0.25)")]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, row=3, col=1)

    # --- Row 4: MACD ---------------------------------------------------------
    macd, signal, hist = _calc_macd(close)
    hist_colors = ["#ef5350" if h < 0 else "#26a69a" for h in hist]

    fig.add_trace(
        go.Bar(
            x=df.index, y=hist, name="MACD Hist",
            marker_color=hist_colors, showlegend=False,
        ),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=macd, name="MACD",
            mode="lines", line=dict(color="#2196F3", width=1.3),
        ),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=signal, name="Signal",
            mode="lines", line=dict(color="#FF9800", width=1.3),
        ),
        row=4, col=1,
    )

    # --- Layout --------------------------------------------------------------
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=900,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=80, b=40),
    )
    fig.update_yaxes(title_text="Price",  row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI",    row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD",   row=4, col=1)

    return fig


# ---------------------------------------------------------------------------
# Telegram signal chart — dark investing.com-style single-panel chart
# ---------------------------------------------------------------------------

# ── Colour palette (investing.com dark theme) ─────────────────────────────
_C = dict(
    bg        = "#131722",   # outer background
    panel     = "#1e222d",   # chart area background
    grid      = "rgba(255,255,255,0.06)",
    text      = "#b2b5be",
    up        = "#26a69a",   # teal-green candles
    down      = "#ef5350",   # red candles
    ma50      = "#FFB74D",   # amber
    ma200     = "#f06292",   # soft pink/magenta
    cur_price = "#9e9e9e",   # current price dashed line
    buy       = "#26a69a",
    target    = "#a5d6a7",   # light green
    stop      = "#ef5350",
)


def build_telegram_chart(
    df: pd.DataFrame,
    ticker: str,
    trade_rec=None,
    score: float | None = None,
    n_days: int = 42,   # ≈ 2 calendar months
) -> go.Figure:
    """
    Dark-theme single-panel chart for Telegram signal posts (investing.com style).

    Shows last n_days candles + MA50 + MA200 + trade zones.
    MAs are computed on the full history so MA200 is accurate.
    """
    # ── MAs on full history, slice display window ─────────────────────────
    close_full = df["Close"]
    ma50_full  = close_full.rolling(50).mean()
    ma200_full = close_full.rolling(200).mean()

    df    = df.tail(n_days).copy()
    ma50  = ma50_full.tail(n_days)
    ma200 = ma200_full.tail(n_days)
    close = df["Close"]
    current_price = float(close.iloc[-1])
    x_start = str(df.index[0])
    x_end   = str(df.index[-1])

    fig = go.Figure()

    # ── Zone fills (drawn first, behind everything) ───────────────────────
    if trade_rec is not None:
        if trade_rec.limit_price:
            fig.add_hrect(
                y0=trade_rec.limit_price * 0.985,
                y1=trade_rec.limit_price * 1.015,
                fillcolor="rgba(38,166,154,0.10)",
                line_width=0, layer="below",
            )
        if trade_rec.stop_price:
            fig.add_hrect(
                y0=trade_rec.stop_price * 0.97,
                y1=trade_rec.stop_price,
                fillcolor="rgba(239,83,80,0.08)",
                line_width=0, layer="below",
            )

    # ── Candlestick ───────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=close,
        name=ticker,
        increasing_line_color=_C["up"],
        decreasing_line_color=_C["down"],
        increasing_fillcolor=_C["up"],
        decreasing_fillcolor=_C["down"],
        showlegend=False,
        line_width=1,
    ))

    # ── MA lines ──────────────────────────────────────────────────────────
    if ma50.notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=ma50,
            name="MA50",
            mode="lines",
            line=dict(color=_C["ma50"], width=1.5, dash="dot"),
        ))
    if ma200.notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=ma200,
            name="MA200",
            mode="lines",
            line=dict(color=_C["ma200"], width=2),
        ))

    # ── Current price hairline ────────────────────────────────────────────
    fig.add_shape(
        type="line",
        x0=x_start, x1=x_end,
        y0=current_price, y1=current_price,
        line=dict(color=_C["cur_price"], width=1, dash="dot"),
        layer="below",
    )
    fig.add_annotation(
        x=x_end, y=current_price,
        text=f"${current_price:.2f}",
        showarrow=False, xanchor="right",
        font=dict(color="white", size=11, family="monospace"),
        bgcolor=_C["panel"],
        bordercolor=_C["cur_price"],
        borderwidth=1,
        borderpad=3,
    )

    # ── Trade level lines ─────────────────────────────────────────────────
    def _hline(y: float, color: str, dash: str, label: str) -> None:
        fig.add_shape(
            type="line",
            x0=x_start, x1=x_end,
            y0=y, y1=y,
            line=dict(color=color, width=1.5, dash=dash),
            layer="above",
        )
        fig.add_annotation(
            x=x_end, y=y,
            text=label,
            showarrow=False, xanchor="right",
            font=dict(color=color, size=11),
            bgcolor=_C["panel"],
            bordercolor=color,
            borderwidth=1,
            borderpad=3,
        )

    if trade_rec is not None:
        if trade_rec.stop_price:
            _hline(trade_rec.stop_price, _C["stop"], "dash",
                   f"Stop ${trade_rec.stop_price:.0f}")
        if trade_rec.limit_price:
            pct = (trade_rec.limit_price / current_price - 1) * 100
            _hline(trade_rec.limit_price, _C["buy"], "dashdot",
                   f"Buy ${trade_rec.limit_price:.0f} ({pct:+.1f}%)")
        if trade_rec.target_price:
            upside = (trade_rec.target_price / current_price - 1) * 100
            _hline(trade_rec.target_price, _C["target"], "dot",
                   f"Target ${trade_rec.target_price:.0f} (+{upside:.0f}%)")

    # ── Y-axis range — based on candle action, not extreme trade levels ───
    candle_lo = float(df["Low"].min())
    candle_hi = float(df["High"].max())
    candle_span = candle_hi - candle_lo

    # Pull nearby levels (limit/stop) into view; cap extreme ones (target far away)
    CLIP_RATIO = 0.20   # don't extend view more than 20% of candle span for any level
    close_levels = [
        v for v in [
            trade_rec.stop_price  if trade_rec else None,
            trade_rec.limit_price if trade_rec else None,
        ] if v is not None
    ]
    all_levels = [
        v for v in [
            trade_rec.stop_price   if trade_rec else None,
            trade_rec.limit_price  if trade_rec else None,
            trade_rec.target_price if trade_rec else None,
        ] if v is not None
    ]

    price_lo = candle_lo
    price_hi = candle_hi
    for lv in close_levels:
        price_lo = min(price_lo, lv)
        price_hi = max(price_hi, lv)
    # Clamp so target can't push the range more than CLIP_RATIO beyond candle range
    price_lo = max(price_lo, candle_lo - candle_span * CLIP_RATIO)
    price_hi = min(price_hi, candle_hi + candle_span * CLIP_RATIO)

    # Ensure all_levels lines are at least visible (clip annotations but keep shapes)
    if all_levels:
        price_lo = min(price_lo, min(all_levels))
        price_hi = max(price_hi, max(all_levels))
        # Re-apply clip after including all levels
        price_lo = max(price_lo, candle_lo - candle_span * CLIP_RATIO)
        price_hi = min(price_hi, candle_hi + candle_span * CLIP_RATIO)

    # Symmetric 4% padding around final range
    pad = (price_hi - price_lo) * 0.04
    price_lo -= pad
    price_hi += pad

    # ── Title ─────────────────────────────────────────────────────────────
    score_str  = f"  ·  Score {score:.0f}/100" if score is not None else ""
    title_text = f"<b>{ticker}</b>  ·  1D  ·  2 months{score_str}"

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_C["bg"],
        plot_bgcolor=_C["panel"],
        height=520,
        width=1100,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="left",   x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=_C["text"]),
        ),
        title=dict(
            text=title_text,
            font=dict(size=13, color=_C["text"]),
            x=0.01, y=0.97,
        ),
        margin=dict(l=10, r=10, t=55, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor=_C["grid"],
            color=_C["text"],
            linecolor="rgba(255,255,255,0.08)",
            tickfont=dict(size=11),
            range=[x_start, x_end],
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=_C["grid"],
            color=_C["text"],
            linecolor="rgba(255,255,255,0.08)",
            range=[price_lo, price_hi],
            tickprefix="$",
            side="right",
            tickfont=dict(size=11),
        ),
    )

    return fig

