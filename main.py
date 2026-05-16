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

from dotenv import load_dotenv

load_dotenv()

from src.engine.engine import analyse
from src.output.formatter import format_brief, format_report


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
    sys.stdout.buffer.write((format_report(result) + "\n").encode("utf-8", errors="replace"))


def cmd_multi(tickers: list[str]) -> None:
    results = [analyse(t) for t in tickers]
    results.sort(key=lambda r: r.overall_score, reverse=True)
    lines = [f"\n{'TICKER':<6} | {'TYPE':<20} | SCORE | SHORT  MED  LONG | RECOMMENDATION",
             "-" * 75]
    for r in results:
        lines.append(format_brief(r))
    lines.append("")
    sys.stdout.buffer.write("\n".join(lines).encode("utf-8", errors="replace"))


def cmd_batch(args: argparse.Namespace, tickers: list[str] | None = None) -> None:
    from src.ai.connector import build_connector
    from src.output.telegram_bot import TelegramBot
    from src.runner import print_batch_summary, run_universe

    if tickers is None:
        tickers = _load_tickers_from_file(args.batch)
    if not tickers:
        print("No tickers found.")
        sys.exit(1)

    # Build AI connector — auto-detect Claude from env if --ai not set
    backend = args.ai
    if backend == "null" and not args.no_ai:
        api_key_env = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key_env:
            backend = "claude"
            print("AI: auto-detected ANTHROPIC_API_KEY → using claude (haiku)")

    kwargs: dict = {}
    if backend == "claude":
        kwargs["api_key"] = args.ai_key or os.environ.get("ANTHROPIC_API_KEY", "")
        # Default to Haiku; allow override via --ai-model
        kwargs["model"] = args.ai_model or "claude-haiku-4-5"
    elif backend == "ollama":
        if args.ai_model:
            kwargs["model"] = args.ai_model
        if args.ai_url:
            kwargs["base_url"] = args.ai_url

    connector = build_connector(backend, **kwargs)

    # Build Telegram bot if requested
    telegram_bot = None
    if args.telegram:
        token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_OUTPUT_CHAT_ID", "")
        if not token or not chat_id:
            print("WARNING: --telegram set but TELEGRAM_BOT_TOKEN / TELEGRAM_OUTPUT_CHAT_ID not found in env.")
        else:
            telegram_bot = TelegramBot(token=token, chat_id=chat_id)
            print(f"Telegram: enabled → chat {chat_id}")

    print(f"\nBatch run: {len(tickers)} tickers  |  AI backend: {backend}  |  min_score: {args.min_score}")

    batch = run_universe(
        tickers=tickers,
        connector=connector,
        output_dir=args.out,
        min_score=args.min_score,
        save_all_reports=args.save_all,
        save_charts=not args.no_charts,
        telegram_bot=telegram_bot,
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
                        help="AI backend (default: auto — uses claude if ANTHROPIC_API_KEY is set)")
    parser.add_argument("--no-ai", action="store_true",
                        help="Disable AI even if ANTHROPIC_API_KEY is in env")
    parser.add_argument("--ai-model", metavar="MODEL",
                        help="Model name for claude or ollama (default for claude: claude-3-5-haiku-20241022)")
    parser.add_argument("--ai-key", metavar="KEY",
                        help="API key for claude (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--ai-url", metavar="URL",
                        help="Base URL for ollama (default: http://localhost:11434)")

    # Telegram options
    parser.add_argument("--telegram", action="store_true",
                        help="Send passing signals to Telegram (reads TELEGRAM_BOT_TOKEN and TELEGRAM_OUTPUT_CHAT_ID from env)")

    args = parser.parse_args()

    if args.batch:
        cmd_batch(args)
    elif args.tickers and (args.telegram or args.ai != "null" or
                           (not args.no_ai and os.environ.get("ANTHROPIC_API_KEY"))):
        # Single/multi with AI or Telegram — route through full pipeline
        cmd_batch(args, tickers=args.tickers)
    elif len(args.tickers) == 1:
        cmd_single(args.tickers[0])
    elif len(args.tickers) > 1:
        cmd_multi(args.tickers)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

