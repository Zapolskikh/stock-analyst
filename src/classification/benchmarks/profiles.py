"""
Benchmark profiles: expected ranges and weights for each stock type.

IMPORTANT — these thresholds are initial reasonable defaults derived from
public equity research norms.  They MUST be validated and calibrated via:
  1. Back-testing on historical data
  2. Comparison with sector medians for each time period
  3. Possibly ML-based calibration (Phase 2)

TODO (managed in manage_todo_list):
  - Verify Pharma / Biotech thresholds (pipeline weight, patent cliff risk)
  - Verify Financial sector metrics (D/E ratio is meaningless for banks;
    need NIM, Tier-1 capital, efficiency ratio instead)
  - Add Turnaround sub-type thresholds
  - Add Emerging-Market adjustments
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Benchmark:
    """
    Per-type scoring weights and acceptable ranges.

    score_weights: relative importance of each block for Overall Score.
    thresholds:    min/ok/good values for key metrics in this type.
                   Values are normalised (same units as StockData).
    """
    name:           str
    description:    str
    score_weights:  dict[str, float]   # must sum to 1.0
    thresholds:     dict[str, dict]    # metric → {min, ok, good}


# ---------------------------------------------------------------------------
# Individual benchmark definitions
# ---------------------------------------------------------------------------

HYPERGROWTH_TECH = Benchmark(
    name="Hypergrowth Tech",
    description="High-growth software/semi/platform. Revenue >20% YoY, high margin expansion.",
    score_weights={
        "quality":   0.35,
        "valuation": 0.20,
        "technical": 0.15,
        "risk":      0.20,
        "style_fit": 0.10,
    },
    thresholds={
        "revenue_growth_yoy": {"min": 0.15, "ok": 0.25, "good": 0.40},
        "gross_margin":       {"min": 0.50, "ok": 0.65, "good": 0.75},
        "operating_margin":   {"min": 0.00, "ok": 0.10, "good": 0.25},
        "fcf_margin":         {"min": 0.00, "ok": 0.10, "good": 0.20},
        "debt_to_equity":     {"min": None, "ok": 1.0,  "good": 0.5},  # lower=better
        # TODO: add Rule-of-40 threshold
    },
)

MATURE_TECH = Benchmark(
    name="Mature Tech",
    description="Established tech with stable earnings, buybacks, moderate growth.",
    score_weights={
        "quality":   0.30,
        "valuation": 0.30,
        "technical": 0.15,
        "risk":      0.15,
        "style_fit": 0.10,
    },
    thresholds={
        "revenue_growth_yoy": {"min": 0.03, "ok": 0.08, "good": 0.15},
        "gross_margin":       {"min": 0.40, "ok": 0.55, "good": 0.70},
        "operating_margin":   {"min": 0.10, "ok": 0.20, "good": 0.30},
        "fcf_yield":          {"min": 0.02, "ok": 0.04, "good": 0.07},
        "pe":                 {"min": None, "ok": 25,   "good": 18},   # lower=better
    },
)

PHARMA_HEALTHCARE = Benchmark(
    name="Pharma / Healthcare",
    description="Drug companies, med devices. Pipeline, patent safety, dividend stability.",
    score_weights={
        "quality":   0.30,
        "valuation": 0.25,
        "technical": 0.10,
        "risk":      0.25,   # patent cliff & regulatory risk elevated
        "style_fit": 0.10,
    },
    thresholds={
        "revenue_growth_yoy": {"min": 0.00, "ok": 0.05, "good": 0.12},
        "gross_margin":       {"min": 0.50, "ok": 0.65, "good": 0.75},
        "fcf_yield":          {"min": 0.03, "ok": 0.05, "good": 0.08},
        "dividend_yield":     {"min": 0.01, "ok": 0.025,"good": 0.04},
        "debt_to_equity":     {"min": None, "ok": 0.8,  "good": 0.4},
        # TODO: add pipeline-quality proxy (R&D% of revenue, # phase-3 drugs)
    },
)

DIVIDEND_DEFENSIVE = Benchmark(
    name="Dividend / Defensive",
    description="Utilities, consumer staples, REITs. Stable cash flow, safe dividend.",
    score_weights={
        "quality":   0.25,
        "valuation": 0.30,
        "technical": 0.10,
        "risk":      0.25,
        "style_fit": 0.10,
    },
    thresholds={
        "dividend_yield":     {"min": 0.025,"ok": 0.035,"good": 0.05},
        "payout_ratio":       {"min": None, "ok": 0.70, "good": 0.50},  # lower=better
        "debt_to_equity":     {"min": None, "ok": 1.0,  "good": 0.5},
        "fcf_yield":          {"min": 0.03, "ok": 0.05, "good": 0.07},
        "revenue_growth_yoy": {"min": -0.02,"ok": 0.02, "good": 0.06},
    },
)

CYCLICAL = Benchmark(
    name="Cyclical",
    description="Energy, materials, industrials. Cycle timing, balance sheet resilience.",
    score_weights={
        "quality":   0.25,
        "valuation": 0.25,
        "technical": 0.20,   # cycle timing matters more
        "risk":      0.20,
        "style_fit": 0.10,
    },
    thresholds={
        "ev_ebitda":          {"min": None, "ok": 8,    "good": 5},    # lower=better
        "debt_to_equity":     {"min": None, "ok": 0.8,  "good": 0.4},
        "fcf_yield":          {"min": 0.04, "ok": 0.07, "good": 0.12},
        # TODO: add cycle-phase signal (e.g. commodity price momentum)
    },
)

FINANCIAL = Benchmark(
    name="Financial",
    description="Banks, insurance. Traditional ratios largely inapplicable; use ROE, P/B.",
    score_weights={
        "quality":   0.35,
        "valuation": 0.30,
        "technical": 0.10,
        "risk":      0.20,
        "style_fit": 0.05,
    },
    thresholds={
        "roe":            {"min": 0.08, "ok": 0.12, "good": 0.18},
        "pe":             {"min": None, "ok": 12,   "good": 8},    # lower=better
        "dividend_yield": {"min": 0.01, "ok": 0.025,"good": 0.04},
        # TODO: NIM (net interest margin), Tier-1 capital ratio, efficiency ratio
        #       These are NOT in yfinance.info — need FMP or manual parsing.
    },
)

TURNAROUND = Benchmark(
    name="Turnaround",
    description="Recovering companies. FCF improvement, debt reduction, margin recovery.",
    score_weights={
        "quality":   0.25,
        "valuation": 0.25,
        "technical": 0.20,
        "risk":      0.25,
        "style_fit": 0.05,
    },
    thresholds={
        "operating_margin":   {"min": -0.05,"ok": 0.05, "good": 0.12},
        "debt_to_equity":     {"min": None, "ok": 2.0,  "good": 1.0},
        "revenue_growth_yoy": {"min": -0.05,"ok": 0.03, "good": 0.10},
        # TODO: need YoY margin-improvement signal, not just absolute value
    },
)

OTHER = Benchmark(
    name="Other",
    description="Catch-all. Uses balanced default weights.",
    score_weights={
        "quality":   0.30,
        "valuation": 0.25,
        "technical": 0.15,
        "risk":      0.20,
        "style_fit": 0.10,
    },
    thresholds={},
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARKS: dict[str, Benchmark] = {
    "hypergrowth_tech":   HYPERGROWTH_TECH,
    "mature_tech":        MATURE_TECH,
    "pharma_healthcare":  PHARMA_HEALTHCARE,
    "dividend_defensive": DIVIDEND_DEFENSIVE,
    "cyclical":           CYCLICAL,
    "financial":          FINANCIAL,
    "turnaround":         TURNAROUND,
    "other":              OTHER,
}


def get(stock_type: str) -> Benchmark:
    return BENCHMARKS.get(stock_type, OTHER)
