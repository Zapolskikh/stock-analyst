"""
Multi-universe pullback scanner with persistent resume state.

Scans configured universes in order.  On the first confirmed signal the
ticker is sent to Telegram, the position is saved to a JSON state file,
and the script exits.  The next run continues from the ticker AFTER the
last signal within the same universe, then moves on to the next one.

State file : data/scan_state.json
Usage      : uv run python scripts/scan_all.py
             uv run python scripts/scan_all.py --reset   # clear state, start over
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
from src.output.telegram_bot import TelegramBot, send_batch_results, send_error_alert
from src.runner import BatchResult

# ---------------------------------------------------------------------------
# Configuration — edit to taste
# ---------------------------------------------------------------------------

STATE_FILE = ROOT / "data" / "scan_state.json"

#: Universes to scan in order.
#: Supported: "sp500", "russell1000", "finviz_undervalued"
SCAN_UNIVERSES: list[str] = ["sp500", "finviz_undervalued"]

#: Tickers that already received a Telegram signal — skip forever.
#: Pre-populated with tickers signalled in prior sessions.
INITIAL_SIGNALED: list[str] = ["ADBE", "META"]

# MIN_SCORE, ANOMALY_LOW, PULLBACK_MIN_PCT, PULLBACK_PERIOD imported from scan_config.py

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
ANTHROPIC_KEY      = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_TOKEN     = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_OUTPUT_CHAT_ID", "")
# Monitoring alerts go to TELEGRAM_MONITOR_CHAT_ID; fallback to output chat.
TELEGRAM_MONITOR_CHAT_ID = os.environ.get("TELEGRAM_MONITOR_CHAT_ID", TELEGRAM_CHAT_ID)

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

_DEFAULT_STATE: dict = {
    "universes":            SCAN_UNIVERSES,
    "resume_universe":      SCAN_UNIVERSES[0],
    "resume_after_ticker":  None,    # None = start from the beginning of that universe
    "signaled":             INITIAL_SIGNALED,
}


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            for k, v in _DEFAULT_STATE.items():
                state.setdefault(k, v)
            return state
        except Exception as exc:
            print(f"  ⚠️  Could not read {STATE_FILE.name} ({exc}), using defaults")
    return dict(_DEFAULT_STATE)


def _save_state(state: dict) -> None:
    state["last_saved"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _reset_state() -> dict:
    state = dict(_DEFAULT_STATE)
    _save_state(state)
    return state


# ---------------------------------------------------------------------------
# Defensive invariant check
# ---------------------------------------------------------------------------

def _expect(condition: bool, message: str) -> None:
    """Явная замена assert для production-кода.

    В отличие от assert не отключается флагом -O и поднимает RuntimeError
    с понятным сообщением, которое дойдёт до алёрта в Telegram.
    """
    if not condition:
        raise RuntimeError(f"Invariant violated: {message}")


# ---------------------------------------------------------------------------
# Pullback helper
# ---------------------------------------------------------------------------

def _check_pullback(ticker: str) -> tuple[bool, float, float, float]:
    """Return (passes, pullback_pct, current_price, period_high)."""
    try:
        from src.data.price import fetch_ohlcv
        df       = fetch_ohlcv(ticker, period=PULLBACK_PERIOD)
        current  = float(df["Close"].iloc[-1])
        high     = float(df["High"].max())
        pullback = (high - current) / high * 100.0
        return pullback >= PULLBACK_MIN_PCT, round(pullback, 1), round(current, 2), round(high, 2)
    except Exception:
        return True, 0.0, 0.0, 0.0   # fetch failed → let through


# ---------------------------------------------------------------------------
# Universe scan
# ---------------------------------------------------------------------------

def _scan_universe(
    uni_name: str,
    label: str,
    tickers: list[str],
    start_after: str | None,
    signaled: set[str],
    connector,
    telegram_bot,
    state: dict,
) -> tuple[bool, list[dict], list[str]]:
    """Run Phase 1 + Phase 2 for one universe.

    Returns (stop, anomalies, signals_found):
      stop           — True means halt the outer universe loop (STOP_ON_FIRST=True)
      anomalies      — borderline candidates (score between ANOMALY_LOW and MIN_SCORE)
      signals_found  — tickers that passed and were sent to Telegram
    """
    # ── determine active slice ────────────────────────────────────────────
    if start_after is not None and start_after not in tickers:
        print(f"  ⚠️  Resume ticker {start_after!r} not found in {label} — scanning from beginning")
        start_after = None

    started = (start_after is None)
    active: list[str] = []
    for t in tickers:
        if not started:
            if t == start_after:
                started = True
            continue
        if t not in signaled:
            active.append(t)

    if not active:
        print(f"  All tickers in {label} already scanned/signalled.")
        return False, []

    if MAX_TICKERS:
        active = active[:MAX_TICKERS]
        print(f"  ⚠️  Capped at {MAX_TICKERS} tickers (MAX_TICKERS debug limit)")

    if start_after:
        print(f"  Resuming after: {start_after!r}  ({len(active)} tickers remaining)")
    else:
        print(f"  Scanning {len(active)} tickers")
    print()

    # ── Phase 1: pullback pre-screen ──────────────────────────────────────
    print(f"Phase 1 — Pullback pre-screen  [{label}]")
    print("─" * 70)

    # (ticker, pullback_pct, current, high)
    shortlist: list[tuple[str, float, float, float]] = []

    for idx, ticker in enumerate(active, 1):
        passes, pullback, current, high = _check_pullback(ticker)
        if passes:
            tag = f"✓  {pullback:.1f}%  (${current:.0f} vs ${high:.0f} peak)"
            shortlist.append((ticker, pullback, current, high))
        else:
            tag = f"✗  {pullback:.1f}% — near peak, skip"
        print(f"  [{idx:03d}/{len(active)}] {ticker:<6}  {tag}")

    print()
    print(f"  Shortlist ({len(shortlist)}): {[t for t, *_ in shortlist]}")
    print()

    if not shortlist:
        print(f"  No pullback candidates in {label}.")
        return False, [], []

    # ── Phase 2: full scoring + AI ────────────────────────────────────────
    print(f"Phase 2 — Full analysis  [{label}]")
    print("─" * 70)
    anomalies:     list[dict] = []
    signals_found: list[str]  = []

    for idx, (ticker, pullback_pct, price_now, price_high) in enumerate(shortlist, 1):
        lbl = f"[{idx:02d}/{len(shortlist)}] {ticker:<6}  pullback={pullback_pct:.1f}%"
        print(f"{lbl}", end="  ", flush=True)

        try:
            result = analyse(ticker)
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        # ── Invariant checks on analysis result ───────────────────────────
        _expect(result is not None, f"analyse({ticker!r}) returned None")
        _expect(
            0.0 <= result.overall_score <= 100.0,
            f"overall_score={result.overall_score} out of [0, 100] for {ticker}",
        )
        _expect(
            isinstance(result.block_scores, dict),
            f"block_scores is not a dict for {ticker}: {type(result.block_scores)}",
        )

        score    = result.overall_score
        decision = result.decision

        if score < MIN_SCORE:
            print(f"score={score:.1f}  {decision:<18}  → SKIPPED")
            if score >= ANOMALY_LOW:
                bs        = result.block_scores
                fv_est    = result.fair_value.fair_value    if result.fair_value else None
                fv_upside = result.fair_value.discount_pct if result.fair_value else None
                key_notes: list[str] = []
                for _bn in ("quality", "valuation", "risk"):
                    _blk = bs.get(_bn)
                    if _blk is not None:
                        key_notes.extend(_blk.notes)
                anomalies.append({
                    "ticker":       ticker,
                    "score":        score,
                    "pullback_pct": pullback_pct,
                    "price":        price_now,
                    "decision":     decision,
                    "universe":     label,
                    "fv_est":       fv_est,
                    "fv_upside":    fv_upside,
                    "key_notes":    key_notes[:6],
                })
            continue

        ai_review = None
        if USE_AI:
            print(f"score={score:.1f}  {decision:<18}  → AI...", end=" ", flush=True)
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
                print(f"AI:✓  conf={ai_review.confidence:.2f}")
            else:
                print(f"AI:✗  conf={ai_review.confidence:.2f}")
        else:
            print(f"score={score:.1f}  {decision:<18}  → AI:skipped")

        # ── Trade rec veto: engine says limit order is impractical ──────────
        if result.trade_rec and result.trade_rec.action == "Avoid":
            reason = result.trade_rec.rationale[0] if result.trade_rec.rationale else "impractical entry"
            print(f"  → VETOED  ({reason})")
            continue

        # ── Signal: post regardless of AI verdict ─────────────────────────
        print()
        print("─" * 70)
        print(f"  SIGNAL: {ticker}  [{label}]")
        print(f"  Score    : {score:.1f}/100  |  Pullback: {pullback_pct:.1f}% off {PULLBACK_PERIOD} high")
        print(f"  Price    : ${price_now:.0f}  (peak ${price_high:.0f})")
        if result.trade_rec and result.trade_rec.limit_price:
            tr = result.trade_rec
            print(f"  Trade    : Buy ${tr.limit_price:.0f}  →  Target ${tr.target_price:.0f}"
                  f"  |  Stop ${tr.stop_price:.0f}")
        if ai_review and not ai_review.abstain and ai_review.narrative:
            verdict = "agrees" if ai_review.agreement else "disagrees"
            print(f"  AI ({verdict}) : {ai_review.narrative[:220]}")
        print("─" * 70)

        sys.stdout.buffer.write(format_report(result).encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")

        # Telegram
        if telegram_bot and not DRY_RUN:
            br = BatchResult(
                ticker=ticker, result=result, error=None,
                ai_review=ai_review, source=label,
            )
            try:
                sent = send_batch_results(telegram_bot, [br], send_chart=True, force=True)
                print(f"  📨 Telegram sent ({sent}) → {TELEGRAM_CHAT_ID}")
            except Exception as exc:
                print(f"  ❌ Telegram error: {exc}")
        elif DRY_RUN:
            print("  📭 [DRY RUN] Telegram send skipped")

        # ── persist state ─────────────────────────────────────────────────
        signaled.add(ticker)
        signals_found.append(ticker)
        state["signaled"]            = sorted(signaled)
        state["resume_universe"]     = uni_name
        state["resume_after_ticker"] = ticker
        _save_state(state)
        print(f"  💾 State saved  →  next run continues after {ticker!r} in {label}")

        if STOP_ON_FIRST:
            if anomalies:
                print()
                print("  Borderline misses (this universe):")
                for a in anomalies:
                    print(_fmt_anomaly(a))
            return True, anomalies, signals_found  # stop the outer loop

        # STOP_ON_FIRST=False — continue scanning the rest of this universe

    # universe exhausted (or all processed when STOP_ON_FIRST=False)
    if anomalies:
        print()
        print("  Borderline misses:")
        for a in anomalies:
            print(_fmt_anomaly(a))
    return False, anomalies, signals_found


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _fmt_anomaly(a: dict) -> str:
    """Форматирует запись о пограничном кандидате для консоли."""
    fv    = a.get("fv_est")
    fv_str = f"  fv=${fv:.0f}" if fv else ""
    return (
        f"  ⚠️  [{a['ticker']}] score={a['score']:.1f}  pullback={a['pullback_pct']:.1f}%"
        f"  ${a.get('price', 0):.0f}{fv_str}  {a.get('decision', '')}"
    )


def _format_summary_post(
    all_signals: list[str],
    all_anomalies: list[dict],
    universes: list[str],
    start_ts: datetime,
) -> str:
    """Формирует HTML-пост для Telegram с итогами скана."""
    import html as _html

    _PERIOD_MAP = {"6mo": "6мес", "3mo": "3мес", "1y": "1г", "2y": "2г"}
    period_lbl  = _PERIOD_MAP.get(PULLBACK_PERIOD, PULLBACK_PERIOD)

    ts_str  = start_ts.strftime("%d %b %Y, %H:%M")
    uni_str = " · ".join(universes)

    lines = [
        f"📊 <b>Скан завершён</b> — {ts_str}",
        f"<i>{_html.escape(uni_str)}</i>",
    ]

    if all_signals:
        tickers_str = ", ".join(all_signals)
        lines.append(f"\n✅ Отправлено сигналов: {len(all_signals)}  ({_html.escape(tickers_str)})")

    if all_anomalies:
        sorted_a = sorted(all_anomalies, key=lambda x: x["score"], reverse=True)
        lines += [
            "",
            "━━━━━━━━━━━━━━━━━━",
            f"⚠️ <b>Почти прошли порог</b>  (score {ANOMALY_LOW:.0f}–{MIN_SCORE:.0f})",
            "━━━━━━━━━━━━━━━━━━",
        ]
        for a in sorted_a[:20]:
            t      = a["ticker"]
            s      = a["score"]
            pb     = a["pullback_pct"]
            px     = a["price"]
            uni    = _html.escape(a.get("universe", ""))
            fv     = a.get("fv_est")
            upside = a.get("fv_upside")
            if fv and upside is not None:
                sign    = "+" if upside >= 0 else ""
                fv_part = f"  ·  оценка ${fv:.0f} ({sign}{upside:.1f}%)"
            else:
                fv_part = ""
            lines.append(f"\n<b>{t}</b>  Score: {s:.1f}/100  ·  {uni}")
            lines.append(f"  Pullback ↓{pb:.1f}% от {period_lbl} хая  ·  ${px:.0f}{fv_part}")
            notes = a.get("key_notes", [])
            for note in notes[:4]:
                lines.append(f"  · {_html.escape(note)}")

    if not all_signals and not all_anomalies:
        lines.append("\n🔍 Кандидатов не найдено")

    return "\n".join(lines)


def _send_summary(
    telegram_bot,
    all_signals: list[str],
    all_anomalies: list[dict],
    universes: list[str],
    start_ts: datetime,
) -> None:
    """Отправляет итоговый пост в Telegram (не бросает исключений)."""
    if telegram_bot is None or DRY_RUN:
        if DRY_RUN:
            post = _format_summary_post(all_signals, all_anomalies, universes, start_ts)
            print("  📭 [DRY RUN] Summary post:")
            print(post)
        return
    try:
        post = _format_summary_post(all_signals, all_anomalies, universes, start_ts)
        telegram_bot.send_text(post)
        print(f"  📨 Summary sent → {TELEGRAM_CHAT_ID}")
    except Exception as exc:
        print(f"  ❌ Summary Telegram error: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-universe pullback scanner")
    parser.add_argument("--reset", action="store_true",
                        help="Clear saved state and start from scratch")
    args = parser.parse_args()

    try:
        _run(args)
    except KeyboardInterrupt:
        print("\n  ⏹  Interrupted by user.")
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"\n  ❌ FATAL: {exc}\n{tb}", flush=True)
        send_error_alert(
            token=TELEGRAM_TOKEN,
            chat_id=TELEGRAM_MONITOR_CHAT_ID,
            message=f"scan_all.py упал с ошибкой:\n\n{tb}",
        )
        raise


def _run(args) -> None:
    """Основной скан. Обёрнут в main(), чтобы планировщик мог поймать Exception."""
    start_ts = datetime.now()  # для итогового поста (локальное время контейнера = ET)

    connector    = (build_connector("claude", api_key=ANTHROPIC_KEY)
                    if ANTHROPIC_KEY else build_connector("null"))
    telegram_bot = (TelegramBot(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)
                    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID else None)

    if args.reset:
        state = _reset_state()
        print("  ✅  State reset.\n")
    else:
        state = _load_state()

    universes: list[str] = state["universes"]
    signaled:  set[str]  = set(state["signaled"])
    resume_uni           = state["resume_universe"]
    resume_after         = state.get("resume_after_ticker")

    print("=" * 70)
    print("  MULTI-UNIVERSE PULLBACK SCANNER")
    print(f"  Universes  : {universes}")
    print(f"  Resume     : {resume_uni}  /  after {resume_after!r}")
    print(f"  Signalled  : {sorted(signaled)}")
    print(f"  AI         : {'claude (haiku)' if ANTHROPIC_KEY else 'null'}{'  [DISABLED]' if not USE_AI else ''}")
    print(f"  Telegram   : {'on → ' + TELEGRAM_CHAT_ID if telegram_bot else 'off'}{'  [DRY RUN]' if DRY_RUN else ''}")
    print(f"  Min score  : {MIN_SCORE}")
    print(f"  Stop mode  : {'first signal' if STOP_ON_FIRST else 'scan all'}")
    if MAX_TICKERS:
        print(f"  Max tickers: {MAX_TICKERS}  (debug limit)")
    print("=" * 70)
    print()

    # index of universe to start from
    resume_idx    = universes.index(resume_uni) if resume_uni in universes else 0
    all_anomalies: list[dict] = []
    all_signals:   list[str]  = []

    for uni_idx, uni_name in enumerate(universes):
        # skip universes we've already fully processed in previous runs
        if uni_idx < resume_idx:
            print(f"  [{uni_idx+1}/{len(universes)}] {uni_name:<22} — already completed, skipping")
            continue

        print(f"\n{'=' * 70}")
        print(f"  Universe [{uni_idx+1}/{len(universes)}]: {uni_name}")

        try:
            tickers, label = fetch_universe(uni_name)
            print(f"  {label}: {len(tickers)} tickers loaded")
            _expect(isinstance(tickers, list) and len(tickers) > 0,
                    f"fetch_universe({uni_name!r}) returned empty ticker list")
        except Exception as exc:
            print(f"  ⚠️  Failed to load {uni_name}: {exc}")
            continue

        # first universe in this run may have a resume position;
        # later universes always start from the beginning
        start_after = resume_after if uni_idx == resume_idx else None

        stop, uni_anomalies, uni_signals = _scan_universe(
            uni_name=uni_name,
            label=label,
            tickers=tickers,
            start_after=start_after,
            signaled=signaled,
            connector=connector,
            telegram_bot=telegram_bot,
            state=state,
        )
        all_anomalies.extend(uni_anomalies)
        all_signals.extend(uni_signals)

        # per-universe Telegram summary
        _send_summary(telegram_bot, all_signals=uni_signals, all_anomalies=uni_anomalies,
                      universes=[uni_name], start_ts=start_ts)

        if stop:
            print()
            print("=" * 70)
            print(f"  🛑  STOPPED — signal found in {label}")
            print("=" * 70)
            return

        # universe exhausted without signal — advance resume to next universe
        if uni_idx + 1 < len(universes):
            state["resume_universe"]     = universes[uni_idx + 1]
            state["resume_after_ticker"] = None
            _save_state(state)
            print(f"  {label} exhausted — advancing to {universes[uni_idx + 1]}")
        else:
            # all universes done
            state["resume_universe"]     = universes[0]
            state["resume_after_ticker"] = None
            _save_state(state)

    # ── all universes processed ───────────────────────────────────────────
    print()
    print("=" * 70)
    if all_signals:
        print(f"  ✅  Run complete — {len(all_signals)} signal(s) sent: {all_signals}")
    else:
        print("  No signals found across all universes.")
        print(f"  Next run will start from the beginning of {state['resume_universe']}.")
    print("=" * 70)


if __name__ == "__main__":
    main()
