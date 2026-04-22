"""
Human-readable output formatter.

Takes an AnalysisResult and returns a formatted text report.
This is the final stage before AI narrative generation and Telegram posting
(both out of scope for this phase).
"""

from __future__ import annotations
from src.engine.engine import AnalysisResult


def format_report(result: AnalysisResult) -> str:
    sd  = result.stock_data
    h   = result.horizons
    bs  = result.block_scores

    lines = [
        "=" * 60,
        f"  STOCK ANALYSIS REPORT",
        "=" * 60,
        f"  Ticker  : {result.ticker}",
        f"  Name    : {sd.name}",
        f"  Sector  : {sd.sector}  |  {sd.industry}",
        f"  Type    : {result.stock_type.replace('_', ' ').title()}",
        "-" * 60,
        f"  OVERALL SCORE   : {result.overall_score:.1f} / 100  →  {result.rating}",
        f"  RECOMMENDATION  : {result.recommendation}",
        "-" * 60,
        "  BLOCK SCORES (0–10)",
        f"    Quality    : {bs.get('quality',   0):.2f}",
        f"    Valuation  : {bs.get('valuation', 0):.2f}",
        f"    Technical  : {bs.get('technical', 0):.2f}",
        f"    Risk       : {bs.get('risk',      0):.2f}",
        f"    Style Fit  : {bs.get('style_fit', 0):.2f}",
        "-" * 60,
        "  HORIZON SCORES (0–100)",
        f"    Short-term  (days–weeks) : {h.short:.1f}",
        f"    Medium-term (months)     : {h.medium:.1f}",
        f"    Long-term   (1–5 years)  : {h.long:.1f}",
        "-" * 60,
        "  KEY METRICS",
        f"    Price        : ${sd.price:.2f}" if sd.price else "    Price        : N/A",
        f"    Market Cap   : ${sd.market_cap/1e9:.1f}B" if sd.market_cap else "    Market Cap   : N/A",
        f"    Rev Growth   : {(sd.revenue_growth_yoy or 0)*100:.1f}%",
        f"    Gross Margin : {(sd.gross_margin or 0)*100:.1f}%",
        f"    Op Margin    : {(sd.operating_margin or 0)*100:.1f}%",
        f"    FCF Yield    : {(sd.fcf_yield or 0)*100:.1f}%",
        f"    P/E (fwd)    : {sd.forward_pe:.1f}" if sd.forward_pe else "    P/E (fwd)    : N/A",
        f"    D/E Ratio    : {sd.debt_to_equity:.2f}" if sd.debt_to_equity else "    D/E Ratio    : N/A",
        "-" * 60,
    ]

    if result.stop_factors:
        lines.append("  ⚠  STOP FACTORS")
        for sf in result.stop_factors:
            lines.append(f"    [{sf.severity.upper()}] {sf.name}")
            lines.append(f"           {sf.description}")
        lines.append("-" * 60)

    lines.append("  RATIONALE")
    for bullet in result.rationale:
        lines.append(f"    • {bullet}")

    lines += ["=" * 60, ""]
    return "\n".join(lines)


def format_brief(result: AnalysisResult) -> str:
    """One-liner summary suitable for list views."""
    h = result.horizons
    return (
        f"{result.ticker:6s} | {result.stock_type:20s} | "
        f"Score: {result.overall_score:5.1f} | "
        f"S:{h.short:4.1f} M:{h.medium:4.1f} L:{h.long:4.1f} | "
        f"{result.recommendation}"
    )
