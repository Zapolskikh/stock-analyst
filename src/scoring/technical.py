"""
Block C — Technical State Score (0–10).

Вопрос: как ведёт себя акция на рынке прямо сейчас?

Метрики (из plan.md):
  price vs MA50 / MA200   — позиция относительно скользящих
  momentum_3m / 6m / 12m — % изменение цены
  drawdown_from_high      — просадка от максимума (%)
  trend_quality           — доля дней выше MA50 (за последние 3 мес)

Данные: nd.close_prices  (последние ≤252 торговых дня, новее = правее)
"""
from __future__ import annotations

import math
from statistics import mean

from src.data.normalizer import NormalisedData
from src.scoring.base import BlockScore, avg_scores


# Minimum bars needed for each indicator
_MIN_MA50  = 50
_MIN_MA200 = 200
_MIN_MOM3  = 63    # ~3 trading months
_MIN_MOM6  = 126
_MIN_MOM12 = 252


def _sma(prices: list[float], period: int) -> float | None:
    """Simple moving average of the last *period* prices, or None if not enough data."""
    tail = [p for p in prices[-period:] if math.isfinite(p)]
    return mean(tail) if len(tail) >= period * 0.9 else None  # allow up to 10% gaps


def _momentum(prices: list[float], lookback: int) -> float | None:
    """% price change from *lookback* bars ago to most recent."""
    valid = [p for p in prices if math.isfinite(p)]
    if len(valid) < lookback + 1:
        return None
    past = valid[-(lookback + 1)]
    now  = valid[-1]
    if past <= 0:
        return None
    return (now - past) / past * 100.0


def _drawdown_from_high(prices: list[float], window: int = 252) -> float | None:
    """% drawdown from the highest close in the last *window* bars (negative number)."""
    valid = [p for p in prices[-window:] if math.isfinite(p)]
    if not valid:
        return None
    peak = max(valid)
    current = valid[-1]
    if peak <= 0:
        return None
    return (current - peak) / peak * 100.0  # ≤ 0


def _trend_quality(prices: list[float], window: int = 63) -> float | None:
    """
    Fraction of the last *window* closes that are above their own trailing MA50,
    returned as a percentage 0–100.
    Used as a proxy for trend consistency.
    """
    tail = prices[-window:]
    if len(tail) < 10:
        return None
    above = 0
    counted = 0
    for i in range(len(tail)):
        # Use whatever history is available for each sub-MA
        sub = prices[: -(window - i)] if (window - i) > 0 else prices
        sub = [p for p in sub if math.isfinite(p)]
        if len(sub) < 10:
            continue
        sub_ma = mean(sub[-min(50, len(sub)):])
        if math.isfinite(tail[i]) and tail[i] > sub_ma:
            above += 1
        counted += 1
    return above / counted * 100.0 if counted else None


# ---------------------------------------------------------------------------
# Scoring helpers (inline thresholds — no Benchmark dependency for technicals)
# ---------------------------------------------------------------------------

def _score_price_vs_ma(price: float, ma: float) -> float:
    """
    Score based on how far price is above/below MA (as % deviation).
    Slightly above MA = good (10); far above = extended (7); below = weak (2–4).
    """
    if not (math.isfinite(price) and math.isfinite(ma) and ma > 0):
        return float("nan")
    pct = (price - ma) / ma * 100.0
    # pct: -20→0, -5→4, 0→7, 5→9, 15→10, 40→7  (extended upside lowers score slightly)
    pts = [(-20, 0), (-5, 4), (0, 7), (5, 9), (15, 10), (40, 7)]
    return _interp(pts, pct)


def _score_momentum(pct: float) -> float:
    """Score momentum: large positive = 10, negative = low."""
    pts = [(-30, 0), (-10, 2), (0, 4), (10, 6), (25, 8), (50, 10)]
    return _interp(pts, pct)


def _score_drawdown(dd: float) -> float:
    """Score drawdown (dd ≤ 0): small drawdown = good."""
    # dd: 0→10, -10→8, -25→5, -50→1
    pts = [(-50, 1), (-25, 5), (-10, 8), (0, 10)]
    return _interp(pts, dd)


def _score_trend_quality(tq: float) -> float:
    """Score trend quality (0–100 % of days above MA)."""
    pts = [(0, 0), (30, 2), (50, 5), (70, 8), (85, 10)]
    return _interp(pts, tq)


def _interp(points: list[tuple[float, float]], value: float) -> float:
    if not math.isfinite(value) or not points:
        return float("nan")
    if value <= points[0][0]:
        return points[0][1]
    if value >= points[-1][0]:
        return points[-1][1]
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x0 <= value <= x1:
            t = (value - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return float("nan")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_technical(nd: NormalisedData) -> BlockScore:
    """
    Compute Technical State score from nd.close_prices.

    Does NOT depend on Benchmark — technical scoring is type-agnostic
    (trend/momentum/drawdown are universal signals).
    """
    prices = nd.close_prices
    breakdown: dict[str, float] = {}
    notes: list[str] = []

    if len(prices) < 10:
        return BlockScore(
            score=5.0,  # neutral when no data
            breakdown={},
            notes=["no price history — defaulting to neutral technical score"],
        )

    current = prices[-1]

    # --- Price vs MA50 -----------------------------------------------------
    ma50 = _sma(prices, 50)
    if ma50 is not None:
        s = _score_price_vs_ma(current, ma50)
        if math.isfinite(s):
            breakdown["price_vs_ma50"] = s
            pct = (current - ma50) / ma50 * 100
            if pct < -10:
                notes.append(f"price {pct:.0f}% below MA50 (weak)")
            elif pct > 20:
                notes.append(f"price {pct:.0f}% above MA50 (extended)")

    # --- Price vs MA200 ----------------------------------------------------
    ma200 = _sma(prices, 200)
    if ma200 is not None:
        s = _score_price_vs_ma(current, ma200)
        if math.isfinite(s):
            breakdown["price_vs_ma200"] = s
            pct = (current - ma200) / ma200 * 100
            if pct < 0:
                notes.append(f"price below MA200 — bearish structure")
            elif pct > 0:
                notes.append(f"price above MA200 — bullish structure")

    # --- Momentum ----------------------------------------------------------
    for label, lookback in [("momentum_3m", _MIN_MOM3),
                             ("momentum_6m", _MIN_MOM6),
                             ("momentum_12m", _MIN_MOM12)]:
        mom = _momentum(prices, lookback)
        if mom is not None:
            s = _score_momentum(mom)
            if math.isfinite(s):
                breakdown[label] = s

    # --- Drawdown from 52-week high ----------------------------------------
    dd = _drawdown_from_high(prices, 252)
    if dd is not None:
        s = _score_drawdown(dd)
        if math.isfinite(s):
            breakdown["drawdown"] = s
            if dd < -40:
                notes.append(f"drawdown {dd:.0f}% from 52w high — deep decline")
            elif dd > -5:
                notes.append("near 52-week high")

    # --- Trend quality (% days above MA50 over last 3m) --------------------
    tq = _trend_quality(prices)
    if tq is not None:
        s = _score_trend_quality(tq)
        if math.isfinite(s):
            breakdown["trend_quality"] = s

    # --- Relative Strength vs SPY -----------------------------------------
    # RS = stock_momentum_3m − spy_momentum_3m
    # Positive = outperforming market, negative = underperforming
    spy_prices = nd.spy_close_prices
    if spy_prices and len(spy_prices) >= _MIN_MOM3 + 1:
        spy_mom = _momentum(spy_prices, _MIN_MOM3)
        stock_mom = _momentum(prices, _MIN_MOM3)
        if spy_mom is not None and stock_mom is not None:
            rs = stock_mom - spy_mom
            # RS +10% → 10, +3% → 8, 0% → 6, -5% → 4, -15% → 2, -30% → 0
            pts = [(-30, 0), (-15, 2), (-5, 4), (0, 6), (3, 8), (10, 10)]
            s = _interp(pts, rs)
            if math.isfinite(s):
                breakdown["relative_strength"] = s
                if rs > 10:
                    notes.append(f"outperforming SPY by {rs:.0f}% (3m)")
                elif rs < -10:
                    notes.append(f"underperforming SPY by {abs(rs):.0f}% (3m)")

    final = avg_scores(breakdown)
    if not breakdown:
        final = 5.0
        notes.append("insufficient price data — neutral score")

    return BlockScore(score=final, breakdown=breakdown, notes=notes)
