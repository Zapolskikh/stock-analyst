"""Block D — Risk Score (0–10).  10 = lowest risk."""
from __future__ import annotations
from src.data.models.stock_data import StockData
from src.scoring._base import score_metric, average


def score(sd: StockData) -> float:
    """
    Risk proxies available from yfinance:
      - debt_to_equity          (lower = less risk)
      - interest_coverage       (higher = less risk)
      - fcf_margin              (positive FCF = safety buffer)
      - volatility_annualised   (lower = less risk)
      - beta                    (lower = less market risk)
      - drawdown_52w            (less severe = lower risk)

    Missing risk dimensions (not available in yfinance):
      - customer concentration
      - litigation / regulatory overhang
      - patent cliff proximity (Pharma)
      - geographic concentration
      - earnings quality (accruals ratio)

    TODO: add accruals ratio as earnings quality signal
    TODO: flag pharma patent-cliff via pipeline data (needs FMP or manual)
    TODO: add governance score proxy (if available from data provider)
    """
    s = [
        # Debt risk: lower D/E is better
        score_metric(sd.debt_to_equity,
                     min_val=None, ok_val=1.0, good_val=0.3,
                     higher_is_better=False),

        # Interest coverage: higher is safer
        score_metric(sd.interest_coverage,
                     min_val=1.5, ok_val=5.0, good_val=15.0),

        # FCF buffer: positive FCF reduces distress risk
        score_metric(sd.fcf_margin,
                     min_val=-0.05, ok_val=0.05, good_val=0.15),

        # Volatility: lower annualised vol = less risk
        score_metric(sd.volatility_annualised,
                     min_val=None, ok_val=0.40, good_val=0.20,
                     higher_is_better=False),

        # Beta: lower beta = less systematic risk
        score_metric(sd.beta,
                     min_val=None, ok_val=1.5, good_val=0.7,
                     higher_is_better=False),
    ]
    return round(average(s), 2)
