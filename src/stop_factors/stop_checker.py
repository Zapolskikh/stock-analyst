"""
Stop-factor checks — hard disqualifiers that override a good score.

Even a high-scoring stock can be flagged AVOID or WATCH if one or more
stop-factors trigger.  These are intentionally conservative.

Returns a list of triggered StopFactor objects.
If any stop-factor has severity='hard', the recommendation moves to 'Avoid'.
If severity='soft', the recommendation is downgraded one level.

TODO: add earnings-quality stop (accruals ratio)
TODO: add regulatory-overhang flag (needs news/sentiment data)
TODO: add patent-cliff flag for pharma (needs pipeline data)
TODO: add liquidity stop for micro-cap stocks (avg daily volume < threshold)
"""

from __future__ import annotations
from dataclasses import dataclass
from src.data.models.stock_data import StockData


@dataclass
class StopFactor:
    name:        str
    description: str
    severity:    str   # 'hard' | 'soft'


def check(sd: StockData) -> list[StopFactor]:
    """Return list of triggered stop factors (empty = clean)."""
    flags: list[StopFactor] = []

    # --- Hard stops ---

    # Negative FCF + high leverage = distress risk
    if (sd.fcf is not None and sd.fcf < 0
            and sd.debt_to_equity is not None and sd.debt_to_equity > 2.0):
        flags.append(StopFactor(
            name="Negative FCF + High Debt",
            description=(
                f"FCF is negative while D/E is {sd.debt_to_equity:.1f}. "
                "Potential liquidity risk."
            ),
            severity="hard",
        ))

    # Interest coverage < 1.5× — can't service debt from operations
    if sd.interest_coverage is not None and sd.interest_coverage < 1.5:
        flags.append(StopFactor(
            name="Weak Interest Coverage",
            description=(
                f"Interest coverage is {sd.interest_coverage:.1f}× "
                "(threshold: 1.5×). Debt service at risk."
            ),
            severity="hard",
        ))

    # --- Soft stops ---

    # Extreme overvaluation: forward P/E > 80 without high growth
    if (sd.forward_pe is not None and sd.forward_pe > 80
            and (sd.revenue_growth_yoy or 0) < 0.30):
        flags.append(StopFactor(
            name="Extreme Valuation",
            description=(
                f"Forward P/E {sd.forward_pe:.0f}× with revenue growth "
                f"{(sd.revenue_growth_yoy or 0)*100:.0f}%. "
                "Limited margin of safety."
            ),
            severity="soft",
        ))

    # Long-term trend broken: price > 30% below 200 MA
    if sd.price_vs_200ma is not None and sd.price_vs_200ma < -0.30:
        flags.append(StopFactor(
            name="Long-term Trend Broken",
            description=(
                f"Price is {sd.price_vs_200ma*100:.0f}% below 200-day MA. "
                "Significant downtrend."
            ),
            severity="soft",
        ))

    # Heavy drawdown from 52-week high
    if sd.drawdown_52w is not None and sd.drawdown_52w < -0.50:
        flags.append(StopFactor(
            name="Severe 52-week Drawdown",
            description=(
                f"Stock is {sd.drawdown_52w*100:.0f}% below 52-week high. "
                "Deep value or fundamental deterioration."
            ),
            severity="soft",
        ))

    # High beta + high debt = amplified risk
    if (sd.beta is not None and sd.beta > 2.0
            and sd.debt_to_equity is not None and sd.debt_to_equity > 1.5):
        flags.append(StopFactor(
            name="High Volatility + High Leverage",
            description=(
                f"Beta {sd.beta:.1f} combined with D/E {sd.debt_to_equity:.1f}. "
                "Amplified downside risk."
            ),
            severity="soft",
        ))

    return flags
