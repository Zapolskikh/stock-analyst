# TODO — Stock Analyst Improvement Backlog

Приоритеты: 🔴 Критично / 🟡 Важно / 🟢 Улучшение

---

## 🔴 ВЫСОКИЙ ПРИОРИТЕТ

### 1. Split normalization через info.json
**Статус:** ✅ СДЕЛАНО (частично) — применяется последний сплит из `info.json`.
Отображается в debug_report: `⚠ Split adj.: 10:1 on 2024-06-10`.
Ограничение: только последний сплит. Для множественных сплитов — см. пункт 9.  
**Проблема:** SEC EDGAR хранит исторические shares/EPS в **сырых** (не скорректированных) значениях.
- NVDA: 2021 → 2479M shares (4:1 split), 2024 → 24643M shares (10:1 split)  
- EPS 2019: $6.63 pre-split vs $4.90 post-split 2026 — исторически несравнимы
- Ломает: dilution score, EPS growth trend, PEG, историческую динамику  

**Решение:** В `normalise()` после загрузки shares/EPS — применить ретроактивную
корректировку на накопленный split factor для всех дат до `lastSplitDate`.

```python
# Логика:
split_factor = parse("10:1")  # → 10.0
split_date = pd.Timestamp(info["lastSplitDate"], unit="s")
# Для каждой строки с end < split_date:
#   shares /= split_factor
#   eps    *= split_factor  (EPS уменьшается при сплите)
```

**Важно:** NVDA имел два сплита (4:1 в 2021, 10:1 в 2024) — info.json хранит только
последний. Нужно проверить `sec_raw_facts.json` на наличие `StockSplitRatio` или
детектировать множественные сплиты через скачки >200% в рядах shares.

---

### 2. Historical P/E (5-year average)
**Статус:** ✅ СДЕЛАНО — вычисляется в `normalise()` из ohlcv + EPS.
Добавлены поля `pe_hist_avg`, `pe_hist_high`, `pe_hist_low` в `NormalisedData`.
Отображается в debug_report Step 1 и AI payload Step 10.  
**Источник:** `ohlcv.parquet` (цена конца года) + `sec_eps_diluted.parquet` (годовой EPS)  
**Ограничение:** `ohlcv.parquet` начинается с мая 2021 → получим ~5 лет истории P/E  
**Файл:** `src/data/normalizer.py`  

**Использование:**
- В `valuation block` — сравнивать текущий P/E с историческим средним (relative cheapness)
- В `AI payload` (debug_report Step 10) — "сейчас дороже/дешевле чем обычно?"

```python
# Добавить в NormalisedData:
pe_5yr_avg: Optional[float] = None   # среднее P/E за 5 лет
pe_5yr_high: Optional[float] = None  # максимум (пиковая оценка)
pe_5yr_low: Optional[float] = None   # минимум (кризисная оценка)
```

**Вычисление:** Для каждого года взять цену закрытия на дату 10-K filing + EPS → P/E.

---

## 🟡 СРЕДНИЙ ПРИОРИТЕТ

### 3. Quarterly EPS actuals vs estimates (beat/miss история)
**Статус:** ⚠️ Требует новой выгрузки из yfinance  
**Источник:** `yfinance.Ticker.earnings_dates` → DataFrame с колонками:
- `EPS Estimate`, `Reported EPS`, `Surprise(%)`, `Surprise`

**Нужно добавить в `scripts/fetch_offline_data.py`:**
```python
# Сохранять как:
data/offline/<TICKER>/yf_earnings_history.parquet
```

**Ограничение:** yfinance даёт последние ~8 кварталов. Нужен интернет при выгрузке.

**Использование:**
- **Risk block** — компания которая регулярно бьёт ожидания имеет более низкий
  execution risk; miss × 3 → предупреждение
- **AI payload** — beat/miss streak как качественный сигнал
- Возможный новый суб-score в quality block: `earnings_reliability`

```python
# Добавить в NormalisedData:
earnings_beat_rate: Optional[float] = None  # % кварталов с beat (0.0–1.0)
earnings_avg_surprise_pct: Optional[float] = None  # средний % сюрприза
```

---

### 4. Revenue segments (топ-3 сегмента)
**Статус:** ❌ Недоступно из SEC XBRL structured facts напрямую  
**Проблема:** Сегментная выручка в SEC хранится в **custom company-specific XBRL tags**
(не стандартный us-gaap namespace). Для NVDA — нет `DatacenterRevenue` в XBRL.
Для AAPL — нет `iPhoneRevenue`. Каждая компания использует свои теги.

**Варианты:**
1. **SEC XBRL + custom namespace** — парсить `sec_raw_facts.json` по компании,
   искать ключи с `Segment`, `Geography`, `Product` в названии → очень нестабильно
2. **Ручная разметка** — добавить `data/offline/<TICKER>/segments.json` вручную
   для топ-20 наших тикеров (одноразовая работа, обновлять раз в год)
3. **Macrotrends/Wisesheets scraping** — нужен интернет + парсинг HTML

**Рекомендация:** Вариант 2 (ручная разметка) для нашего набора из 20 тикеров.
Формат:
```json
{
  "segments": [
    {"name": "Compute & Networking", "revenue_pct": 87, "year": 2025},
    {"name": "Graphics", "revenue_pct": 13, "year": 2025}
  ],
  "top_segment_pct": 87,
  "source": "10-K 2025"
}
```

**Использование:** Только в AI payload — concentration risk signal.

---

### 5. Добавить shortRatio + institutionalOwnership в AI payload
**Статус:** ✅ СДЕЛАНО — `short_ratio`, `short_pct_float`, `institutional_ownership`, `insider_ownership`,
`recommendation_key`, `recommendation_mean` добавлены в `NormalisedData`, загружаются в `normalise()`,
`short_ratio` используется в risk block, все поля присутствуют в AI payload (Step 10).

---

## 🟢 НИЗКИЙ ПРИОРИТЕТ / FUTURE

### 6. Confidence system (High / Medium / Low)
Финальная уверенность в анализе на основе:
- `data_quality` (good / partial / poor)
- `model_spread_pct` в fair value (>60% → Low confidence)
- Количество triggered stop factors
- Согласованность block scores

Выводить в header репорта и в trade recommendation.

### 7. Macro regime fields (при наличии online fetch)
- 10yr Treasury yield (актуальный, не константа 4.5%)
- VIX уровень
- Sector P/E relative to S&P 500

### 8. Entry tiers (расширение Trade Recommendation)
Вместо Buy / Skip добавить:
```
Aggressive Buy   — discount > 15%, strong tech confirmation
Buy              — текущее BUY NOW
Accumulate       — текущее BUY ON LIMIT
Watch            — хорошая компания, ждём catalyst
Skip             — текущий Skip
```

### 9. Множественные исторические сплиты
Info.json хранит только **последний** split. Для компаний с 2+ сплитами (AAPL: 2014, 2020;
NVDA: 2021, 2024; TSLA: 2020, 2022) нужно либо:
- Хранить полную историю сплитов в отдельном файле `yf_splits.parquet`
- Или детектировать автоматически через скачки >200% в shares_outstanding

---

## Порядок выполнения

1. **Historical P/E** (пункт 2) — самый быстрый: данные уже есть, только код
2. **Short/institutional в AI payload** (пункт 5) — 10 минут
3. **Split normalization** (пункт 1) — сложнее, требует тестирования на NVDA/AAPL
4. **Quarterly EPS fetch** (пункт 3) — требует обновления fetch_offline_data.py + re-fetch
5. **Revenue segments ручная разметка** (пункт 4) — ручная работа, когда будет время

---

## 🔴 ИНФРАСТРУКТУРА / LIVE-режим

### 10. Актуальность данных — критичный вопрос
**Статус:** ❌ Не решено — нужен регулярный re-fetch

**Проблема:** Offline данные (`data/offline/<TICKER>/`) стареют. Даже если они скачаны
сегодня утром, уже завтра:
- `current_price` из `info.json` — устарел (цена из последней сессии)
- `ohlcv` — нет сегодняшней свечи → технические индикаторы (MA50, MA200) смещены
- Если вышел квартальный отчёт — SEC facts не обновятся сами

**Что нужно:**
1. `fetch_offline_data.py` запускать **перед каждым анализом** или ежедневно по расписанию
2. Для live-режима: цену (`current_price`) и OHLCV брать напрямую из yfinance каждый раз —
   это быстро и бесплатно, не требует SEC API
3. SEC EDGAR (фундаментал) обновлять реже — раз в неделю достаточно,
   т.к. 10-Q выходит раз в квартал

**Разделение по источникам:**
| Данные | Источник | Частота обновления |
|---|---|---|
| Цена, объём, OHLCV | yfinance | Перед каждым запуском |
| Analyst targets, short ratio | yfinance | Ежедневно |
| Фундаментал (revenue, EPS, CF) | SEC EDGAR | Еженедельно / после 10-Q |

**Пока нет доступа к SEC API** — использовать то что есть, но помнить что фундаментал
может быть устаревшим на 1–2 квартала. В `debug_report` в Data Quality Breakdown
уже показывается `dq_historical` — следить за ним.

---

### 11. Telegram-бот — генерация поста для акций прошедших фильтры
**Статус:** ❌ Не начато — планируется позже

**Логика:**
- Запускается `run_batch.py` по списку тикеров
- Акции с `action = "Accumulate"` или `"Accumulate on Pullback"` → формируют пост
- Пост отправляется в Telegram канал
- Отдельным сообщением или inline — вывод от AI (согласие/несогласие + краткое обоснование)

**Формат поста (draft):**
```
🟢 MSFT — Accumulate on Pullback
Score: 72.5/100 | MATURE_TECH

📊 Лимит: $415.00 (−3% от текущей $428.50) | ждать 3 дня
🎯 Цель: $510.00 (+22%) | Горизонт: medium (3–12 мес.)
🛑 Стоп: $372.00 (−10%)

Обоснование: [top rationale lines]

🤖 AI: ✅ Согласен — [краткое обоснование]
     или ❌ Не согласен — [причина]
```

**AI в этой схеме:**
- Получает payload только для акций с `action != "Avoid"`
- Возвращает: `agree: bool` + `rationale: str` (1–2 предложения)
- На первом этапе: посты идут в канал с AI-мнением, решение о покупке — вручную

### 12. AI connector — формат запроса для approved акций
**Статус:** ❌ Не реализовано

**Что нужно:**
- Взять готовый AI payload из Step 10 `debug_report`
- Добавить системный промпт: "ты — second opinion analyst, дай agree/disagree + 2 предложения"
- Парсить ответ: извлечь `agree: bool` и `rationale: str`
- Интегрировать в `run_batch.py` → `post_generator.py` → Telegram
