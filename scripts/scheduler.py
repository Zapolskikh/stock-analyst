"""
Ежедневный планировщик — точка входа для Docker-контейнера.

Запускает scan_all.py один раз в день по рабочим дням в 10:00 ET
(через 30 минут после открытия NYSE в 9:30 ET).

Поведение при старте:
  - Если сегодня будний день, время ≥ 10:00 ET и скан ещё не запускался —
    запуск немедленно (контейнер мог упасть после открытия биржи).
  - Иначе — ожидание до следующего запланированного момента.

Переменные окружения (помимо уже используемых scan_all.py):
  SCAN_RUN_HOUR   — час запуска по ET (по умолчанию 10)
  SCAN_RUN_MINUTE — минута запуска по ET (по умолчанию 0)

Использование:
  uv run python scripts/scheduler.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# ── Пути ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # for scan_config, scan_all

from dotenv import load_dotenv

load_dotenv()

# ── Зависимости (после load_dotenv) ─────────────────────────────────────────
from src.output.telegram_bot import send_error_alert

# ── Конфиг ──────────────────────────────────────────────────────────────────
ET = ZoneInfo("America/New_York")

RUN_HOUR   = int(os.environ.get("SCAN_RUN_HOUR",   "10"))
RUN_MINUTE = int(os.environ.get("SCAN_RUN_MINUTE",  "30"))

TELEGRAM_TOKEN          = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_MONITOR_CHAT_ID = os.environ.get(
    "TELEGRAM_MONITOR_CHAT_ID",
    os.environ.get("TELEGRAM_OUTPUT_CHAT_ID", ""),
)

SCHEDULER_STATE_FILE = ROOT / "data" / "scheduler_state.json"

# Интервал опроса (секунды). 60 с = проверяем раз в минуту.
POLL_INTERVAL = 60

# При ошибке скана — ждать N секунд перед следующей попыткой на следующий день.
ERROR_BACKOFF = 300  # 5 минут


# ---------------------------------------------------------------------------
# Состояние планировщика
# ---------------------------------------------------------------------------

def _load_scheduler_state() -> dict:
    if SCHEDULER_STATE_FILE.exists():
        try:
            return json.loads(SCHEDULER_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_scheduler_state(state: dict) -> None:
    SCHEDULER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULER_STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Вспомогательные функции времени
# ---------------------------------------------------------------------------

def _now_et() -> datetime:
    return datetime.now(ET)


def _run_target_today(now: datetime) -> datetime:
    """Момент запуска для сегодняшнего дня (в ET)."""
    return now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)


def _is_weekday(now: datetime) -> bool:
    return now.weekday() < 5  # Пн–Пт = 0–4


def _has_run_today(state: dict, now: datetime) -> bool:
    last = state.get("last_run_date")
    return last == now.strftime("%Y-%m-%d")


def _seconds_until(target: datetime) -> float:
    return max(0.0, (target - _now_et()).total_seconds())


def _next_weekday_run_target(now: datetime) -> datetime:
    """Следующий момент запуска (завтра или после выходных)."""
    candidate = _run_target_today(now) + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


# ---------------------------------------------------------------------------
# Запуск скана
# ---------------------------------------------------------------------------

def _run_scan() -> None:
    """Вызывает scan_all.main() с аргументами по умолчанию."""
    import argparse

    import scan_all  # в sys.path через scripts/

    # Имитируем вызов без аргументов (--reset не нужен в продакшене)
    scan_all._run(argparse.Namespace(reset=False))


# ---------------------------------------------------------------------------
# Основной цикл
# ---------------------------------------------------------------------------

def run_scheduler() -> None:
    print(f"  Scheduler started — run target {RUN_HOUR:02d}:{RUN_MINUTE:02d} ET on weekdays",
          flush=True)

    while True:
        now   = _now_et()
        state = _load_scheduler_state()

        # ── Выходные — ждём до следующего рабочего дня ────────────────────
        if not _is_weekday(now):
            target = _next_weekday_run_target(now)
            secs   = _seconds_until(target)
            print(f"  [{now.strftime('%Y-%m-%d %H:%M ET')}] Weekend — sleeping until "
                  f"{target.strftime('%Y-%m-%d %H:%M ET')} ({secs/3600:.1f} h)", flush=True)
            time.sleep(min(secs, POLL_INTERVAL))
            continue

        # ── Уже запускался сегодня — ждём завтра ──────────────────────────
        if _has_run_today(state, now):
            target = _next_weekday_run_target(now)
            secs   = _seconds_until(target)
            print(f"  [{now.strftime('%H:%M ET')}] Already ran today — sleeping until "
                  f"{target.strftime('%Y-%m-%d %H:%M ET')} ({secs/3600:.1f} h)", flush=True)
            time.sleep(min(secs, POLL_INTERVAL))
            continue

        # ── Ещё рано — ждём времени запуска ───────────────────────────────
        target = _run_target_today(now)
        if now < target:
            secs = _seconds_until(target)
            if secs > POLL_INTERVAL * 2:
                print(f"  [{now.strftime('%H:%M ET')}] Waiting until "
                      f"{target.strftime('%H:%M ET')} ({secs/60:.0f} min)", flush=True)
            time.sleep(min(secs, POLL_INTERVAL))
            continue

        # ── Время пришло — запускаем скан! ────────────────────────────────
        ts = now.strftime("%Y-%m-%d %H:%M ET")
        print(f"\n  [{ts}] Starting daily scan...", flush=True)

        try:
            _run_scan()
            # Помечаем что сегодня запустились
            state["last_run_date"] = now.strftime("%Y-%m-%d")
            state["last_run_ts"]   = now.isoformat()
            _save_scheduler_state(state)
            print(f"  Scan completed. Next run: {_next_weekday_run_target(_now_et()).strftime('%Y-%m-%d %H:%M ET')}",
                  flush=True)

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            tb = traceback.format_exc()
            print(f"\n  ❌ Scan failed: {exc}\n{tb}", flush=True)
            send_error_alert(
                token=TELEGRAM_TOKEN,
                chat_id=TELEGRAM_MONITOR_CHAT_ID,
                message=f"Ежедневный скан упал:\n\n{tb}",
            )
            # Всё равно помечаем как «запущен», чтобы не ретраить сегодня
            state["last_run_date"]  = now.strftime("%Y-%m-%d")
            state["last_run_ts"]    = now.isoformat()
            state["last_run_error"] = str(exc)
            _save_scheduler_state(state)
            print(f"  Waiting {ERROR_BACKOFF}s before resuming scheduler...", flush=True)
            time.sleep(ERROR_BACKOFF)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_scheduler()
    except KeyboardInterrupt:
        print("\n  ⏹  Scheduler stopped by user.")
