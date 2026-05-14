"""
Stock Analyst — CLI entry point.

Single ticker (full report):
    python main.py NVDA

Multi-ticker quick table:
    python main.py NVDA AAPL MSFT

Batch mode with AI filter (reads tickers from file, saves reports + charts):
    python main.py --batch tickers.txt
    python main.py --batch tickers.txt --ai ollama --ai-model llama3.1
    python main.py --batch tickers.txt --ai claude --ai-key sk-...
    python main.py --batch tickers.txt --min-score 70 --out output/run1

tickers.txt format: one ticker per line, # lines are comments.
"""

from __future__ import annotations
import argparse
import os
import sys

from src.engine.engine    import analyse
from src.output.formatter import format_report, format_brief


def _load_tickers_from_file(path: str) -> list[str]:
    """Read tickers from a file — one per line, # lines are comments."""
    tickers = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                tickers.append(line.upper())
    return tickers


def cmd_single(ticker: str) -> None:
    result = analyse(ticker)
    print(format_report(result))


def cmd_multi(tickers: list[str]) -> None:
    results = [analyse(t) for t in tickers]
    results.sort(key=lambda r: r.overall_score, reverse=True)
    print(f"\n{'TICKER':<6} | {'TYPE':<20} | SCORE | SHORT  MED  LONG | RECOMMENDATION")
    print("-" * 75)
    for r in results:
        print(format_brief(r))
    print()


def cmd_batch(args: argparse.Namespace) -> None:
    from src.ai.connector import build_connector
    from src.runner import run_universe, print_batch_summary

    tickers = _load_tickers_from_file(args.batch)
    if not tickers:
        print("No tickers found in file.")
        sys.exit(1)

    # Build AI connector
    backend = args.ai or "null"
    kwargs: dict = {}
    if backend == "claude":
        kwargs["api_key"] = args.ai_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if args.ai_model:
            kwargs["model"] = args.ai_model
    elif backend == "ollama":
        if args.ai_model:
            kwargs["model"] = args.ai_model
        if args.ai_url:
            kwargs["base_url"] = args.ai_url

    connector = build_connector(backend, **kwargs)

    print(f"\nBatch run: {len(tickers)} tickers  |  AI backend: {backend}  |  min_score: {args.min_score}")

    batch = run_universe(
        tickers=tickers,
        connector=connector,
        output_dir=args.out,
        min_score=args.min_score,
        save_all_reports=args.save_all,
        save_charts=not args.no_charts,
    )

    print_batch_summary(batch)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stock Analyst — single, multi, or batch mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional: zero or more tickers for single/multi mode
    parser.add_argument("tickers", nargs="*", help="Ticker symbol(s) for single/multi mode")

    # Batch mode
    parser.add_argument("--batch", metavar="FILE",
                        help="Path to a file with tickers (one per line) for batch mode")
    parser.add_argument("--out", default="output/batch",
                        help="Output directory for batch reports and charts (default: output/batch)")
    parser.add_argument("--min-score", type=float, default=65.0,
                        help="Minimum overall score to pass to AI filter (default: 65)")
    parser.add_argument("--save-all", action="store_true",
                        help="Save reports for all tickers, not just those passing the screen")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip saving price charts")

    # AI options
    parser.add_argument("--ai", choices=["null", "claude", "ollama"], default="null",
                        help="AI backend (default: null — skip AI)")
    parser.add_argument("--ai-model", metavar="MODEL",
                        help="Model name for claude or ollama backend")
    parser.add_argument("--ai-key", metavar="KEY",
                        help="API key for claude (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--ai-url", metavar="URL",
                        help="Base URL for ollama (default: http://localhost:11434)")

    args = parser.parse_args()

    if args.batch:
        cmd_batch(args)
    elif len(args.tickers) == 1:
        cmd_single(args.tickers[0])
    elif len(args.tickers) > 1:
        cmd_multi(args.tickers)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

