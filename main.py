"""
Stock Analyst — CLI entry point.

Usage:
    python main.py NVDA
    python main.py NVDA AAPL MSFT
"""

from __future__ import annotations
import sys

from src.engine.engine    import analyse
from src.output.formatter import format_report, format_brief


def main(tickers: list[str]) -> None:
    if len(tickers) == 1:
        result = analyse(tickers[0])
        print(format_report(result))
    else:
        results = [analyse(t) for t in tickers]
        results.sort(key=lambda r: r.overall_score, reverse=True)
        print(f"\n{'TICKER':<6} | {'TYPE':<20} | SCORE | SHORT  MED  LONG | RECOMMENDATION")
        print("-" * 75)
        for r in results:
            print(format_brief(r))
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py TICKER [TICKER ...]")
        sys.exit(1)
    main(sys.argv[1:])
