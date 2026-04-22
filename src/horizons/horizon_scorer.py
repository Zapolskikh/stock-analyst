"""
Horizon scorer — produces short / medium / long-term scores.

Each horizon uses a different weighting of the 5 blocks
(see config/settings.py HORIZON_WEIGHTS).
"""

from __future__ import annotations
from dataclasses import dataclass
from config.settings import HORIZON_WEIGHTS


@dataclass
class HorizonScores:
    short:  float   # 0–100 %
    medium: float
    long:   float

    def as_dict(self) -> dict[str, float]:
        return {"short": self.short, "medium": self.medium, "long": self.long}


def compute(block_scores: dict[str, float]) -> HorizonScores:
    """
    Parameters
    ----------
    block_scores : dict with keys quality, valuation, technical, risk, style_fit
                   Values in [0, 10].

    Returns
    -------
    HorizonScores with values in [0, 100].
    """
    return HorizonScores(
        short  = _weighted(block_scores, HORIZON_WEIGHTS["short"]),
        medium = _weighted(block_scores, HORIZON_WEIGHTS["medium"]),
        long   = _weighted(block_scores, HORIZON_WEIGHTS["long"]),
    )


def _weighted(scores: dict[str, float], weights: dict[str, float]) -> float:
    total_w = sum(weights.values())
    weighted_sum = sum(scores.get(k, 5.0) * w for k, w in weights.items())
    return round((weighted_sum / total_w) * 10, 1)   # scale [0,10] → [0,100]
