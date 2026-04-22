"""Block C — Technical State Score (0–10)."""
from __future__ import annotations
from src.data.models.stock_data import StockData
from src.scoring._base import score_metric, average


def score(sd: StockData) -> float:
    """
    Evaluates momentum, trend quality (vs MA), drawdown, relative strength.

    TODO: add volume-trend confirmation
    TODO: add RSI / MACD regime signal (overbought / oversold)
    TODO: add relative strength vs sector (not just SPY)
    """
    s = [
        # Price vs moving averages: above MA is positive
        score_metric(sd.price_vs_50ma,   min_val=-0.20, ok_val=0.0,  good_val=0.10),
        score_metric(sd.price_vs_200ma,  min_val=-0.30, ok_val=0.0,  good_val=0.15),

        # Momentum: positive momentum is better
        score_metric(sd.momentum_3m,     min_val=-0.20, ok_val=0.0,  good_val=0.15),
        score_metric(sd.momentum_6m,     min_val=-0.25, ok_val=0.0,  good_val=0.20),
        score_metric(sd.momentum_12m,    min_val=-0.30, ok_val=0.05, good_val=0.25),

        # Drawdown from 52w high: small drawdown is better
        # drawdown_52w is negative; less negative = better
        score_metric(sd.drawdown_52w,    min_val=-0.60, ok_val=-0.15, good_val=-0.05),

        # Relative strength vs SPY: positive = outperforming
        score_metric(sd.relative_strength_vs_spy,
                     min_val=-0.30, ok_val=0.0, good_val=0.15),
    ]
    return round(average(s), 2)
