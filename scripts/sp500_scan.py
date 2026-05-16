"""
Temporary S&P 500 top-20 scanner.

Processes tickers one by one in full pipeline mode (AI + Telegram).
Stops after the FIRST ticker that gets a Telegram signal sent.
Flags anomalies where the algorithm may have been too conservative.

Usage:
    uv run python scripts/sp500_scan.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # for scan_config

from dotenv import load_dotenv

load_dotenv()

from scan_config import (
    ANOMALY_LOW,
    DRY_RUN,
    MAX_TICKERS,
    MIN_SCORE,
    STOP_ON_FIRST,
    USE_AI,
)

from src.ai.connector import AIInput, build_connector
from src.engine.engine import analyse
from src.output.formatter import format_report
from src.output.telegram_bot import TelegramBot, send_batch_results
from src.runner import BatchResult

# ---------------------------------------------------------------------------
# S&P 500 expanded scan — top ~40 by market cap (May 2026)
# META at the end — already fired a signal on the previous run
# ---------------------------------------------------------------------------
TICKERS = [
    # ── Top 20 (minus META) ────────────────────────────────────────────────
    "NVDA",   # 1.  Semiconductors / AI infrastructure
    "AAPL",   # 2.  Consumer tech
    "MSFT",   # 3.  Cloud / enterprise
    "AMZN",   # 4.  E-commerce + AWS
    "GOOGL",  # 5.  Search + cloud
    "TSLA",   # 6.  EV / energy
    "AVGO",   # 7.  Semiconductors
    "LLY",    # 8.  Pharma / GLP-1
    "JPM",    # 9.  Banking
    "V",      # 10. Payments
    "UNH",    # 11. Healthcare
    "XOM",    # 12. Energy
    "MA",     # 13. Payments
    "COST",   # 14. Retail
    "HD",     # 15. Home improvement
    "NFLX",   # 16. Streaming
    "JNJ",    # 17. Pharma / medtech
    "PG",     # 18. Consumer staples
    "BAC",    # 19. Banking
    # ── Extended universe ─────────────────────────────────────────────────
    "WMT",    # 20. Retail / consumer
    "ORCL",   # 21. Cloud / enterprise
    "PM",     # 22. Tobacco / consumer staples
    "WFC",    # 23. Banking
    "GS",     # 24. Investment banking
    "MS",     # 25. Wealth mgmt / banking
    "ABBV",   # 26. Pharma / immunology
    "AMD",    # 27. Semiconductors
    "QCOM",   # 28. Semiconductors / wireless
    "TXN",    # 29. Analog semiconductors
    "INTU",   # 30. Fintech / SaaS
    "CRM",    # 31. Enterprise SaaS
    "ADBE",   # 32. Creative / enterprise SaaS
    "NOW",    # 33. IT workflow SaaS
    "PANW",   # 34. Cybersecurity
    "MU",     # 35. Memory semiconductors
    "AMAT",   # 36. Semiconductor equipment
    "CAT",    # 37. Industrial / construction
    "DE",     # 38. Agriculture / industrial
    "RTX",    # 39. Defense / aerospace
    # ── Already fired — do not stop on this ───────────────────────────────
    "META",   # 40. Social / AR  [previous signal — validation run]
]

# MIN_SCORE / ANOMALY_LOW imported from scan_config.py
SKIP_SIGNAL_FOR = {"META"}  # already fired — run for validation, don't stop

# ---------------------------------------------------------------------------
# Setup AI connector
# ---------------------------------------------------------------------------
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
connector = build_connector("claude", api_key=ANTHROPIC_KEY) if ANTHROPIC_KEY else build_connector("null")

# Setup Telegram
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_OUTPUT_CHAT_ID", "")
telegram_bot = None
if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
    telegram_bot = TelegramBot(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)

print("=" * 70)
print("  S&P 500 EXPANDED SCANNER  (stop-on-first-signal mode)")
print(f"  AI backend : {'claude (haiku)' if ANTHROPIC_KEY else 'null'}{'  [DISABLED]' if not USE_AI else ''}")
print(f"  Telegram   : {'enabled → ' + TELEGRAM_CHAT_ID if telegram_bot else 'disabled'}{'  [DRY RUN]' if DRY_RUN else ''}")
print(f"  Min score  : {MIN_SCORE}")
print(f"  Stop mode  : {'first signal' if STOP_ON_FIRST else 'scan all'}")
if MAX_TICKERS:
    print(f"  Max tickers: {MAX_TICKERS}  (debug limit)")
print("=" * 70)
print()

anomalies: list[str] = []
passed_count = 0

_tickers_to_scan = TICKERS[:MAX_TICKERS] if MAX_TICKERS else TICKERS
for idx, ticker in enumerate(_tickers_to_scan, 1):
    print(f"[{idx:02d}/{len(TICKERS)}] {ticker:<6}", end="  ", flush=True)

    # ── Analysis ──────────────────────────────────────────────────────────
    try:
        result = analyse(ticker)
    except Exception as exc:
        print(f"ERROR: {exc}")
        continue

    score = result.overall_score
    decision = result.decision

    # ── Below threshold ───────────────────────────────────────────────────
    if score < MIN_SCORE:
        print(f"score={score:.1f}  {decision:<16}  → SKIPPED")

        # Flag borderline cases
        if score >= ANOMALY_LOW:
            bs   = result.block_scores  # dict[str, BlockScore]
            qual = bs["quality"].score   if "quality"   in bs else None
            val  = bs["valuation"].score if "valuation" in bs else None
            tech = bs["technical"].score if "technical" in bs else None
            note = f"  ⚠️  ANOMALY CANDIDATE [{ticker}] score={score:.1f} — borderline"
            if qual is not None: note += f", quality={qual:.1f}"
            if val  is not None: note += f", valuation={val:.1f}"
            if tech is not None and tech < 3.0 and qual is not None and qual >= 7.0:
                note += f", tech={tech:.1f} ← ONLY DRAG (strong fundamentals, weak tech)"
            anomalies.append(note)
        continue

    # ── Passed threshold: run AI (result posted regardless of verdict) ────
    ai_review = None
    if USE_AI:
        print(f"score={score:.1f}  {decision:<16}  → AI check...", end=" ", flush=True)
        try:
            ai_input  = AIInput.from_result(result)
            ai_review = connector.review(ai_input)
        except Exception as exc:
            print(f"AI ERROR: {exc}")

        if ai_review is None:
            print("AI:error")
        elif ai_review.abstain:
            print("AI:abstain")
        elif ai_review.agreement:
            print(f"AI:✓  confidence={ai_review.confidence:.2f}")
        else:
            print(f"AI:✗  confidence={ai_review.confidence:.2f}")
    else:
        print(f"score={score:.1f}  {decision:<16}  → AI:skipped")

    # ── Score passed — signal regardless of AI verdict ───────────────────
    is_validation = ticker in SKIP_SIGNAL_FOR
    print()
    print("─" * 70)
    print(f"  SIGNAL: {ticker}")
    print(f"  Score    : {score:.1f} / 100")
    print(f"  Decision : {decision}")
    if ai_review and not ai_review.abstain and ai_review.narrative:
        verdict = "agrees" if ai_review.agreement else "disagrees"
        print(f"  AI ({verdict}) : {ai_review.narrative[:200]}")
    print("─" * 70)

    # Print full report
    sys.stdout.buffer.write(format_report(result).encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")

    # Send to Telegram (skip if validation-only ticker or dry run)
    if telegram_bot and not is_validation and not DRY_RUN:
        br = BatchResult(ticker=ticker, result=result, error=None, ai_review=ai_review)
        try:
            sent = send_batch_results(telegram_bot, [br], send_chart=True, force=True)
            print(f"  📨 Telegram sent ({sent}) → {TELEGRAM_CHAT_ID}")
        except Exception as exc:
            print(f"  ❌ Telegram error: {exc}")
    elif DRY_RUN and not is_validation:
        print("  📭 [DRY RUN] Telegram send skipped")

    passed_count += 1

    # ── STOP after first signal ────────────────────────────────────────────
    if not is_validation and STOP_ON_FIRST:
        print()
        print("=" * 70)
        print(f"  🛑  STOPPED after signal on {ticker}  [{idx}/{len(_tickers_to_scan)}]")
        remaining = _tickers_to_scan[idx:]
        if remaining:
            print(f"  Remaining unchecked: {remaining}")
        print("=" * 70)
        break

# ── Summary ───────────────────────────────────────────────────────────────
print()
if passed_count == 0:
    print("  No signals found in this run.")

if anomalies:
    print()
    print("─" * 70)
    print("  ANOMALY NOTES (borderline rejections worth reviewing):")
    for a in anomalies:
        print(a)
    print("─" * 70)
