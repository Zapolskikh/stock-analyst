"""Block B — Valuation Score (0–10).  Higher score = cheaper / more attractive."""
from __future__ import annotations
from src.data.models.stock_data import StockData
from src.scoring._base import score_metric, average


def score(sd: StockData) -> float:
    """
    Uses P/E, forward P/E, EV/EBITDA, P/S, PEG, P/FCF, FCF yield.
    Lower multiples → higher score (except FCF yield: higher = better).

    TODO: incorporate sector-relative percentile so a P/E of 25 in tech
          is scored differently from a P/E of 25 in consumer staples.
    """
    s = [
        # lower is better — pass higher_is_better=False
        score_metric(sd.pe,          min_val=None, ok_val=25, good_val=15,
                     higher_is_better=False),
        score_metric(sd.forward_pe,  min_val=None, ok_val=22, good_val=13,
                     higher_is_better=False),
        score_metric(sd.ev_ebitda,   min_val=None, ok_val=18, good_val=10,
                     higher_is_better=False),
        score_metric(sd.ps,          min_val=None, ok_val=5,  good_val=2,
                     higher_is_better=False),
        score_metric(sd.peg,         min_val=None, ok_val=2,  good_val=1,
                     higher_is_better=False),
        score_metric(sd.p_fcf,       min_val=None, ok_val=25, good_val=15,
                     higher_is_better=False),
        # higher is better
        score_metric(sd.fcf_yield,   min_val=0.0,  ok_val=0.03, good_val=0.07),
    ]
    return round(average(s), 2)
