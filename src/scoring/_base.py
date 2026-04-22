"""
Scoring helpers shared by all blocks.

Each block returns a score in [0, 10].
_score_metric() converts a raw value to 0-10 using min/ok/good thresholds.
"""

from __future__ import annotations
from typing import Optional


def score_metric(value: Optional[float],
                 min_val: Optional[float],
                 ok_val:  float,
                 good_val: float,
                 higher_is_better: bool = True) -> float:
    """
    Map a single metric to [0, 10].

    Segments (higher_is_better=True):
      value < min_val  → 0
      min_val – ok_val → 0–5  (linear)
      ok_val – good_val→ 5–8  (linear)
      > good_val       → 8–10 (capped at 10)

    For lower_is_better metrics pass higher_is_better=False; the value
    is negated before scoring so the same thresholds can be used with
    negated direction values.
    """
    if value is None:
        return 5.0   # neutral / unknown → middle score

    if not higher_is_better:
        value, min_val, ok_val, good_val = (
            -value,
            -min_val if min_val is not None else None,
            -ok_val,
            -good_val,
        )

    if min_val is not None and value < min_val:
        return 0.0
    if value >= good_val:
        extra = (value - good_val) / max(abs(good_val) * 0.5, 0.01)
        return min(10.0, 8.0 + 2.0 * min(extra, 1.0))
    if value >= ok_val:
        frac = (value - ok_val) / (good_val - ok_val)
        return 5.0 + 3.0 * frac
    # between min and ok
    lo = min_val if min_val is not None else ok_val - abs(ok_val)
    if lo == ok_val:
        return 2.5
    frac = (value - lo) / (ok_val - lo)
    return max(0.0, 5.0 * frac)


def average(scores: list[Optional[float]]) -> float:
    """Average ignoring None values. Returns 5 if all None."""
    valid = [s for s in scores if s is not None]
    return sum(valid) / len(valid) if valid else 5.0
