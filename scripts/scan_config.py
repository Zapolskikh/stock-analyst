"""
Shared configuration for all scanner scripts.

Edit this file to tune thresholds and toggle debug behaviour.
All scan scripts (sp500_scan.py, sp500_pullback_scan.py, scan_all.py) import
from here so you only need to change one place.

──────────────────────────────────────────────────────────────────────────────
Примеры запуска из терминала
──────────────────────────────────────────────────────────────────────────────

# Стандартный прогон S&P 500 (стоп на первом сигнале):
uv run python scripts/sp500_scan.py

# Pullback-скан S&P 500 (только акции, упавшие ≥ PULLBACK_MIN_PCT от хая):
uv run python scripts/sp500_pullback_scan.py

# Мульти-юниверс скан с сохранением позиции (продолжает с места остановки):
uv run python scripts/scan_all.py

# Сбросить состояние мульти-юниверс скана и начать заново:
uv run python scripts/scan_all.py --reset

──────────────────────────────────────────────────────────────────────────────
Типичные сценарии отладки  (отредактируй параметры ниже, затем запусти)
──────────────────────────────────────────────────────────────────────────────

# Быстрый тест без AI и без Telegram (только первые 5 тикеров):
#   USE_AI  = False
#   DRY_RUN = True
#   MAX_TICKERS = 5

# Посмотреть все сигналы без остановки:
#   STOP_ON_FIRST = False

# Снизить порог, чтобы увидеть больше кандидатов:
#   MIN_SCORE = 55.0
#   ANOMALY_LOW = 45.0

# Более агрессивный pullback-фильтр (только глубокие коррекции):
#   PULLBACK_MIN_PCT = 15.0
"""

import os

# ── Production flag ──────────────────────────────────────────────────────────

# True when STOCK_ANALYST_ENV=production (set automatically by Docker Compose).
# In production mode: AI call logs are not written to disk (stdout only),
# which prevents unbounded file accumulation in the container.
PRODUCTION: bool = os.environ.get("STOCK_ANALYST_ENV", "dev").lower() == "production"

# ── Scoring thresholds ──────────────────────────────────────────────────────

# Тикер проходит дальше только если его итоговый score ≥ MIN_SCORE.
# Сигнал отправляется в Telegram.  Снизьте до 55–60, чтобы увидеть больше сигналов.
MIN_SCORE   = 65.0

# Тикеры со score в диапазоне [ANOMALY_LOW, MIN_SCORE) не порождают сигнал,
# но выводятся в конце скана как «пограничные» (borderline misses) — полезно
# для ручной проверки случаев, когда алгоритм «почти» дал сигнал.
ANOMALY_LOW = MIN_SCORE - 10.0

# ── Pullback filter (sp500_pullback_scan, scan_all) ─────────────────────────

# Минимальный откат от максимума за PULLBACK_PERIOD.  Тикеры, у которых
# текущая цена ближе к хаю чем на PULLBACK_MIN_PCT %, пропускаются в Phase 1
# (считаются «у пика» — не интересны для стратегии «купить на снижении»).
# Пример: PULLBACK_MIN_PCT = 7.0  →  входим, только если цена упала ≥ 7 % от хая.
PULLBACK_MIN_PCT = 7.0

# Период, за который берётся максимальная цена (reference high).
# "6mo" = последние 6 месяцев; можно поставить "3mo", "1y", "2y".
# Чем больше период — тем «старее» хай и тем больше тикеров пройдут фильтр
# (потому что откат от далёкого пика всегда выглядит крупнее).
PULLBACK_PERIOD  = "6mo"

# ── Scanner behaviour ────────────────────────────────────────────────────────

# True  = остановиться после ПЕРВОГО найденного сигнала (режим по умолчанию).
# False = пройти весь список до конца и собрать все сигналы (полный прогон).
STOP_ON_FIRST = False

# Ограничить скан первыми N тикерами.  None = без ограничения.
# Полезно для быстрой отладки: MAX_TICKERS = 5 прогоняет только первые 5.
MAX_TICKERS   = None

# ── Debug toggles ────────────────────────────────────────────────────────────

# True  = запускать AI-анализ (Claude haiku) после прохождения порога.
# False = пропустить AI-вызов полностью (быстрее, не тратит API-квоту).
USE_AI  = True

# True  = НЕ отправлять сообщения в Telegram (только печатать в консоль).
# False = отправлять в Telegram как обычно.
DRY_RUN = False
