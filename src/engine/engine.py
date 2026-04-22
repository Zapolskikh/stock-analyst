"""
Main scoring engine — orchestrates the full pipeline for one ticker.

Pipeline
--------
fetch → clean → classify → score×5 → horizons → stop-check → final result

Usage
-----
    from src.engine.engine import analyse
    result = analyse("NVDA")
    print(result)
"""

from __future__ import annotations
from dataclasses import dataclass, field

from config.settings import DEFAULT_WEIGHTS, RATING_BANDS, PRICE_HISTORY_DAYS

from src.data.fetchers.yfinance_fetcher import fetch_ticker
from src.data.cleaners.normalizer       import clean
from src.data.models.stock_data         import StockData

from src.classification.classifier      import classify
from src.classification.benchmarks.profiles import get as get_benchmark

import src.scoring.quality   as _quality
import src.scoring.valuation as _valuation
import src.scoring.technical as _technical
import src.scoring.risk      as _risk
import src.scoring.style_fit as _style_fit

from src.horizons.horizon_scorer        import compute as compute_horizons, HorizonScores
from src.stop_factors.stop_checker      import check as check_stops, StopFactor


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    ticker:       str
    stock_type:   str
    stock_data:   StockData

    # Block scores [0, 10]
    block_scores: dict[str, float] = field(default_factory=dict)

    # Overall score [0, 100]
    overall_score: float = 0.0

    # Horizon scores [0, 100]
    horizons: HorizonScores = field(default_factory=lambda: HorizonScores(0, 0, 0))

    # Stop factors
    stop_factors: list[StopFactor] = field(default_factory=list)

    # Final outputs
    rating:         str = ""
    recommendation: str = ""
    rationale:      list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse(ticker: str) -> AnalysisResult:
    """Run the full pipeline for a single ticker."""

    # 1. Fetch
    raw = fetch_ticker(ticker, price_history_days=PRICE_HISTORY_DAYS)

    # 2. Clean & normalise
    sd = clean(ticker, raw)

    # 3. Classify
    stock_type = classify(sd)
    benchmark  = get_benchmark(stock_type)

    # 4. Score 5 blocks
    blocks = {
        "quality":   _quality.score(sd),
        "valuation": _valuation.score(sd),
        "technical": _technical.score(sd),
        "risk":      _risk.score(sd),
        "style_fit": _style_fit.score(sd),
    }

    # 5. Overall score using benchmark weights (fallback to defaults)
    weights = benchmark.score_weights or DEFAULT_WEIGHTS
    overall = sum(blocks[k] * weights.get(k, 0) for k in blocks)
    overall_pct = round(overall * 10, 1)   # [0,10] → [0,100]

    # 6. Horizon scores
    horizons = compute_horizons(blocks)

    # 7. Stop factors
    stops = check_stops(sd)

    # 8. Rating & recommendation
    hard_stops = [s for s in stops if s.severity == "hard"]
    soft_stops = [s for s in stops if s.severity == "soft"]

    rating = _band(overall_pct)

    if hard_stops:
        recommendation = "Avoid"
    elif soft_stops:
        recommendation = _downgrade(_recommend(overall_pct))
    else:
        recommendation = _recommend(overall_pct)

    # 9. Rationale bullets
    rationale = _build_rationale(sd, blocks, stops)

    return AnalysisResult(
        ticker        = ticker.upper(),
        stock_type    = stock_type,
        stock_data    = sd,
        block_scores  = blocks,
        overall_score = overall_pct,
        horizons      = horizons,
        stop_factors  = stops,
        rating        = rating,
        recommendation= recommendation,
        rationale     = rationale,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _band(score: float) -> str:
    for threshold, label in RATING_BANDS:
        if score >= threshold:
            return label
    return "Avoid"


def _recommend(score: float) -> str:
    if score >= 85: return "Buy Now"
    if score >= 70: return "Watch / Accumulate"
    if score >= 55: return "Hold"
    return "Avoid"


_DOWNGRADE_MAP = {
    "Buy Now":           "Watch / Accumulate",
    "Watch / Accumulate":"Hold",
    "Hold":              "Avoid",
    "Avoid":             "Avoid",
}

def _downgrade(rec: str) -> str:
    return _DOWNGRADE_MAP.get(rec, "Hold")


def _build_rationale(sd: StockData,
                     blocks: dict[str, float],
                     stops: list[StopFactor]) -> list[str]:
    r = []

    # Strengths
    if blocks.get("quality", 0) >= 7.5:
        r.append("Strong business quality (revenue growth, margins, returns)")
    if blocks.get("valuation", 0) >= 7.0:
        r.append("Attractive valuation relative to type")
    if blocks.get("technical", 0) >= 7.0:
        r.append("Positive technical momentum")
    if blocks.get("risk", 0) >= 7.0:
        r.append("Low financial risk profile")

    # Weaknesses
    if blocks.get("quality", 10) <= 4.0:
        r.append("Weak business fundamentals")
    if blocks.get("valuation", 10) <= 4.0:
        r.append("Elevated valuation — limited margin of safety")
    if blocks.get("technical", 10) <= 4.0:
        r.append("Bearish technical setup")
    if blocks.get("risk", 10) <= 4.0:
        r.append("Elevated risk (debt, volatility, or FCF)")

    # Stop factors
    for sf in stops:
        r.append(f"[{sf.severity.upper()} STOP] {sf.name}: {sf.description}")

    return r or ["Insufficient distinctive signals for detailed rationale."]
