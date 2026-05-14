"""
Shared types for scoring blocks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BlockScore:
    """Result of a single scoring block (0–10 scale)."""
    score: float                           # final block score 0–10
    breakdown: dict[str, float] = field(default_factory=dict)  # metric → 0–10
    notes: list[str] = field(default_factory=list)             # human-readable

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            self.score = 0.0
        self.score = max(0.0, min(10.0, self.score))


def avg_scores(scores: dict[str, float]) -> float:
    """
    Return the mean of all finite scores in *scores*.
    Returns 0.0 if no finite values.
    """
    vals = [v for v in scores.values() if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else 0.0
