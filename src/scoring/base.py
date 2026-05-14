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
    score: float                           # final block score 0–10 (coverage-adjusted)
    breakdown: dict[str, float] = field(default_factory=dict)  # metric → 0–10
    notes: list[str] = field(default_factory=list)             # human-readable
    coverage: float = 1.0                  # fraction of expected metrics available (0..1)

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            self.score = 0.0
        self.score = max(0.0, min(10.0, self.score))
        self.coverage = max(0.0, min(1.0, self.coverage))


def avg_scores(scores: dict[str, float], expected_count: Optional[int] = None) -> float:
    """
    Return the mean of all finite scores in *scores*.

    Coverage penalty: when *expected_count* is given and exceeds the number of
    finite values, the raw mean is multiplied by sqrt(available / expected).
    This prevents sparse data from artificially inflating block scores — a company
    with 2 out of 7 quality metrics available cannot score the same as one with 7/7.

    Returns 0.0 if no finite values.
    """
    vals = [v for v in scores.values() if math.isfinite(v)]
    if not vals:
        return 0.0
    raw = sum(vals) / len(vals)
    if expected_count and expected_count > 0 and len(vals) < expected_count:
        coverage = len(vals) / expected_count
        raw *= math.sqrt(coverage)
    return raw
