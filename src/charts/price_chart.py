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
