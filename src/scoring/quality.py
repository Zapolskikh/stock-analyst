"""Block A — Business Quality Score (0–10)."""
from __future__ import annotations
from src.data.models.stock_data import StockData
from src.scoring._base import score_metric, average


def score(sd: StockData) -> float:
    """
    Evaluates: revenue growth, EPS growth, ROE, ROIC,
               gross/operating margin, FCF generation, earnings consistency.
    Returns quality_score in [0, 10].
    """
    s = [
        score_metric(sd.revenue_growth_yoy,  min_val=0.0,  ok_val=0.07, good_val=0.20),
        score_metric(sd.eps_growth_yoy,       min_val=0.0,  ok_val=0.07, good_val=0.20),
        score_metric(sd.roe,                  min_val=0.05, ok_val=0.12, good_val=0.20),
        score_metric(sd.roic,                 min_val=0.05, ok_val=0.12, good_val=0.20),
        score_metric(sd.gross_margin,         min_val=0.20, ok_val=0.40, good_val=0.60),
        score_metric(sd.operating_margin,     min_val=0.0,  ok_val=0.10, good_val=0.25),
        score_metric(sd.fcf_margin,           min_val=0.0,  ok_val=0.08, good_val=0.18),
    ]
    return round(average(s), 2)
