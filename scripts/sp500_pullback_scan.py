"""
S&P 500 Pullback Scanner.

Phase 1  — quick price pre-screen: skips tickers that are still near their
           6-month highs (pullback < PULLBACK_MIN_PCT).
Phase 2  — full scoring + AI on the shortlist only.
Phase 3  — first confirmed signal → Telegram (stop-on-first).

Rationale: "buy the dip" — only analyse stocks that have already corrected.

Usage:
    uv run python scripts/sp500_pullback_scan.py
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
    PULLBACK_MIN_PCT,
    PULLBACK_PERIOD,
    STOP_ON_FIRST,
    USE_AI,
)

from src.ai.connector import AIInput, build_connector
from src.data.universe import fetch_universe
from src.engine.engine import analyse
from src.output.formatter import format_report
from src.output.telegram_bot import TelegramBot, send_batch_results
from src.runner import BatchResult

# ---------------------------------------------------------------------------
# Universe — fetched dynamically from live sources (cached 24 h)
# Add "russell1000" to also scan the additional ~500 mid-caps.
# ---------------------------------------------------------------------------
SCAN_UNIVERSES: list[str] = ["sp500"]

# Don't stop the scan on these (already sent to Telegram in previous sessions)
SKIP_SIGNAL_FOR: set[str] = {"ADBE", "META", "MSFT"}

# ---------------------------------------------------------------------------
# Parameters  — from scan_config.py
# ---------------------------------------------------------------------------
# MIN_SCORE, ANOMALY_LOW, PULLBACK_MIN_PCT, PULLBACK_PERIOD imported above

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
ANTHROPIC_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_OUTPUT_CHAT_ID", "")

connector    = (build_connector("claude", api_key=ANTHROPIC_KEY)
                if ANTHROPIC_KEY else build_connector("null"))
telegram_bot = (TelegramBot(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)
                if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID else None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_pullback(ticker: str) -> tuple[bool, float, float, float]:
    """Return (passes, pullback_pct, current_price, period_high).

    *passes* is True when current price is ≥ PULLBACK_MIN_PCT below the
    PULLBACK_PERIOD high.  On fetch failure, returns (True, …) so the ticker
    is not silently dropped.
    """
    try:
        from src.data.price import fetch_ohlcv
        df       = fetch_ohlcv(ticker, period=PULLBACK_PERIOD)
        current  = float(df["Close"].iloc[-1])
        high     = float(df["High"].max())
        pullback = (high - current) / high * 100.0
        return pullback >= PULLBACK_MIN_PCT, round(pullback, 1), round(current, 2), round(high, 2)
    except Exception:
        return True, 0.0, 0.0, 0.0    # can't fetch → let through


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("=" * 70)
print(f"  S&P 500 PULLBACK SCANNER  (≥{PULLBACK_MIN_PCT:.0f}% off {PULLBACK_PERIOD} high)")
print(f"  AI backend : {'claude (haiku)' if ANTHROPIC_KEY else 'null'}{'  [DISABLED]' if not USE_AI else ''}")
print(f"  Telegram   : {'enabled → ' + TELEGRAM_CHAT_ID if telegram_bot else 'disabled'}{'  [DRY RUN]' if DRY_RUN else ''}")
print(f"  Min score  : {MIN_SCORE}")
print(f"  Stop mode  : {'first signal' if STOP_ON_FIRST else 'scan all'}")
if MAX_TICKERS:
    print(f"  Max tickers: {MAX_TICKERS}  (debug limit)")
print("=" * 70)
print()

# ── Build universe from configured sources ─────────────────────────────────
ticker_source: dict[str, str] = {}   # ticker → display label (first source wins)
for _uname in SCAN_UNIVERSES:
    try:
        _tickers, _label = fetch_universe(_uname)
        new_count = sum(1 for t in _tickers if t not in ticker_source)
        for _t in _tickers:
            if _t not in ticker_source:
                ticker_source[_t] = _label
        print(f"  {_label}: {len(_tickers)} tickers ({new_count} new)")
    except Exception as _exc:
        print(f"  ⚠️  Failed to load {_uname}: {_exc}")

if not ticker_source:
    print("  ❌  No tickers loaded — aborting")
    sys.exit(1)

print(f"  Total universe: {len(ticker_source)} tickers")
if MAX_TICKERS:
    ticker_source = dict(list(ticker_source.items())[:MAX_TICKERS])
    print(f"  ⚠️  Capped at {MAX_TICKERS} tickers (MAX_TICKERS debug limit)")
print()

# ── Phase 1: price pre-screen ──────────────────────────────────────────────
print("Phase 1 — Pullback pre-screen")
print("─" * 70)

# (ticker, source, pullback_pct, current, high)
shortlist: list[tuple[str, str, float, float, float]] = []

for idx, ticker in enumerate(ticker_source, 1):
    source = ticker_source[ticker]
    passes, pullback, current, high = _check_pullback(ticker)
    if passes:
        tag = f"✓  {pullback:.1f}%  (${current:.0f} vs ${high:.0f} peak)"
        shortlist.append((ticker, source, pullback, current, high))
    else:
        tag = f"✗  {pullback:.1f}% off high — near peak, skip"
    print(f"  [{idx:03d}/{len(ticker_source)}] {ticker:<6}  {tag}")

print()
print(f"  Shortlist ({len(shortlist)} tickers): {[t for t, *_ in shortlist]}")
print()

if not shortlist:
    print("  No tickers passed the pullback filter.")
    sys.exit(0)

# ── Phase 2: full scoring + AI ────────────────────────────────────────────
print("Phase 2 — Full analysis")
print("─" * 70)

passed_count = 0
anomalies: list[str] = []

for idx, (ticker, source, pullback_pct, price_now, price_high) in enumerate(shortlist, 1):
    label = f"[{idx:02d}/{len(shortlist)}] {ticker:<6}  pullback={pullback_pct:.1f}%  [{source}]"
    print(f"{label}", end="  ", flush=True)

    # Analysis
    try:
        result = analyse(ticker)
    except Exception as exc:
        print(f"ERROR: {exc}")
        continue

    score    = result.overall_score
    decision = result.decision

    if score < MIN_SCORE:
        print(f"score={score:.1f}  {decision:<18}  → SKIPPED")
        if score >= ANOMALY_LOW:
            bs   = result.block_scores
            qual = bs["quality"].score   if "quality"   in bs else None
            val  = bs["valuation"].score if "valuation" in bs else None
            tech = bs["technical"].score if "technical" in bs else None
            note = (f"  ⚠️  [{ticker}] score={score:.1f}  pullback={pullback_pct:.1f}%"
                    f"  qual={qual}  val={val}  tech={tech}")
            anomalies.append(note)
        continue

    # Passed threshold — run AI
    print(f"score={score:.1f}  {decision:<18}  → AI...", end=" ", flush=True)

    ai_review = None
    try:
        ai_input  = AIInput.from_result(result)
        ai_review = connector.review(ai_input)
    except Exception as exc:
        print(f"AI ERROR: {exc}")
        continue

    if ai_review.abstain:
        print("AI:abstain → skip")
        continue

    if not ai_review.agreement:
        print("AI:✗ rejected → skip")
        continue

    # ── Signal confirmed ──────────────────────────────────────────────────
    is_validation = ticker in SKIP_SIGNAL_FOR
    print(f"AI:✓  conf={ai_review.confidence:.2f}{'  [validation]' if is_validation else ''}")
    print()
    print("─" * 70)
    print(f"  ✅  SIGNAL: {ticker}")
    print(f"  Score    : {score:.1f}/100  |  Pullback: {pullback_pct:.1f}% off 6M high")
    print(f"  Price    : ${price_now:.0f}  (peak ${price_high:.0f})")
    print(f"  Source   : {source}")
    print(f"  Decision : {decision}")
    if result.trade_rec and result.trade_rec.limit_price:
        tr = result.trade_rec
        print(f"  Trade    : Buy ${tr.limit_price:.0f}  →  Target ${tr.target_price:.0f}"
              f"  |  Stop ${tr.stop_price:.0f}")
    if ai_review.narrative:
        print(f"  AI note  : {ai_review.narrative[:220]}")
    print("─" * 70)

    # Full report to stdout
    sys.stdout.buffer.write(format_report(result).encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")

    # Telegram
    if telegram_bot and not is_validation and not DRY_RUN:
        br = BatchResult(ticker=ticker, result=result, error=None, ai_review=ai_review, source=source)
        try:
            sent = send_batch_results(telegram_bot, [br], send_chart=True)
            print(f"  📨 Telegram sent ({sent}) → {TELEGRAM_CHAT_ID}")
        except Exception as exc:
            print(f"  ❌ Telegram error: {exc}")
    elif DRY_RUN and not is_validation:
        print("  📭 [DRY RUN] Telegram send skipped")

    passed_count += 1

    # Stop on first non-validation signal
    if not is_validation and STOP_ON_FIRST:
        remaining = [t for t, *_ in shortlist[idx:]]
        print()
        print("=" * 70)
        print(f"  🛑  STOPPED — signal sent for {ticker}  [{idx}/{len(shortlist)}]")
        if remaining:
            print(f"  Remaining pullback candidates: {remaining}")
        print("=" * 70)
        break

# ── Summary ────────────────────────────────────────────────────────────────
print()
if passed_count == 0:
    print("  No signals found among pullback candidates.")

if anomalies:
    print()
    print("─" * 70)
    print("  Borderline misses (score 55–65, pulled back, but just below threshold):")
    for a in anomalies:
        print(a)
    print("─" * 70)
