"""
Block E — Style Fit Score (0–10).

Measures how well the stock matches the expectations of its own type.
Each type has a dedicated scoring function that emphasises the metrics
most relevant to that class.
"""

from __future__ import annotations
from src.data.models.stock_data import StockData
from src.scoring._base import score_metric, average


def score(sd: StockData) -> float:
    dispatch = {
        "hypergrowth_tech":   _hypergrowth_tech,
        "mature_tech":        _mature_tech,
        "pharma_healthcare":  _pharma_healthcare,
        "dividend_defensive": _dividend_defensive,
        "cyclical":           _cyclical,
        "financial":          _financial,
        "turnaround":         _turnaround,
    }
    fn = dispatch.get(sd.stock_type or "other", _generic)
    return round(fn(sd), 2)


# ---------------------------------------------------------------------------
# Per-type style-fit evaluators
# ---------------------------------------------------------------------------

def _hypergrowth_tech(sd: StockData) -> float:
    """Rule-of-40 + high margin + strong momentum."""
    # Rule of 40: revenue_growth% + FCF_margin% >= 40
    rog = (sd.revenue_growth_yoy or 0) * 100 + (sd.fcf_margin or 0) * 100
    s = [
        score_metric(sd.revenue_growth_yoy, min_val=0.15, ok_val=0.25, good_val=0.40),
        score_metric(sd.gross_margin,        min_val=0.50, ok_val=0.65, good_val=0.78),
        score_metric(rog,                    min_val=20,   ok_val=35,   good_val=50),
        score_metric(sd.momentum_12m,        min_val=-0.20,ok_val=0.10, good_val=0.40),
    ]
    return average(s)


def _mature_tech(sd: StockData) -> float:
    s = [
        score_metric(sd.fcf_yield,       min_val=0.02, ok_val=0.04,  good_val=0.07),
        score_metric(sd.operating_margin,min_val=0.10, ok_val=0.20,  good_val=0.30),
        score_metric(sd.roe,             min_val=0.08, ok_val=0.15,  good_val=0.25),
    ]
    return average(s)


def _pharma_healthcare(sd: StockData) -> float:
    """
    TODO: real style fit needs pipeline data (# phase-3 drugs, patent expiry dates).
          Currently uses available proxies only.
    """
    s = [
        score_metric(sd.fcf_yield,      min_val=0.02, ok_val=0.04,  good_val=0.08),
        score_metric(sd.gross_margin,   min_val=0.50, ok_val=0.65,  good_val=0.75),
        score_metric(sd.dividend_yield, min_val=0.01, ok_val=0.025, good_val=0.04),
        score_metric(sd.debt_to_equity, min_val=None, ok_val=0.8,   good_val=0.3,
                     higher_is_better=False),
    ]
    return average(s)


def _dividend_defensive(sd: StockData) -> float:
    s = [
        score_metric(sd.dividend_yield, min_val=0.02,  ok_val=0.03,  good_val=0.05),
        score_metric(sd.payout_ratio,   min_val=None,  ok_val=0.70,  good_val=0.50,
                     higher_is_better=False),
        score_metric(sd.fcf_yield,      min_val=0.03,  ok_val=0.05,  good_val=0.08),
        score_metric(sd.beta,           min_val=None,  ok_val=0.8,   good_val=0.4,
                     higher_is_better=False),
    ]
    return average(s)


def _cyclical(sd: StockData) -> float:
    """
    TODO: cycle-phase signal not yet implemented.
          Needs commodity price momentum or PMI data.
    """
    s = [
        score_metric(sd.ev_ebitda,      min_val=None, ok_val=10, good_val=5,
                     higher_is_better=False),
        score_metric(sd.fcf_yield,      min_val=0.04, ok_val=0.07, good_val=0.12),
        score_metric(sd.debt_to_equity, min_val=None, ok_val=0.8,  good_val=0.3,
                     higher_is_better=False),
    ]
    return average(s)


def _financial(sd: StockData) -> float:
    """
    TODO: NIM, Tier-1 capital, efficiency ratio not in yfinance.
          Current proxy: ROE + P/E only.
    """
    s = [
        score_metric(sd.roe, min_val=0.06, ok_val=0.12, good_val=0.18),
        score_metric(sd.pe,  min_val=None, ok_val=12,   good_val=7,
                     higher_is_better=False),
    ]
    return average(s)


def _turnaround(sd: StockData) -> float:
    """Improvement trend matters more than current level."""
    s = [
        # operating margin becoming positive is the key signal
        score_metric(sd.operating_margin,
                     min_val=-0.15, ok_val=0.0, good_val=0.08),
        score_metric(sd.revenue_growth_yoy,
                     min_val=-0.10, ok_val=0.03, good_val=0.12),
        score_metric(sd.debt_to_equity,
                     min_val=None,  ok_val=2.5,  good_val=1.0,
                     higher_is_better=False),
    ]
    return average(s)


def _generic(sd: StockData) -> float:
    s = [
        score_metric(sd.roe,            min_val=0.05, ok_val=0.12, good_val=0.20),
        score_metric(sd.fcf_margin,     min_val=0.0,  ok_val=0.05, good_val=0.15),
        score_metric(sd.operating_margin,min_val=0.0, ok_val=0.10, good_val=0.22),
    ]
    return average(s)
