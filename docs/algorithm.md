# Stock Analyst — Algorithm Documentation

> Документ для финансового аналитика.  
> Описывает каждый шаг алгоритма: что берётся, как обрабатывается, какие параметры влияют на результат и что можно настроить.

---

## 0. Схема всего пайплайна

```
[Тикер]
   ↓
① Получение сырых данных          src/data/price.py · src/data/sec_edgar.py · src/data/cache.py
   ↓
② Нормализация и очистка           src/data/normalizer.py
   ↓
③ Классификация типа компании      src/classifier.py
   ↓
④ Выбор эталона (бенчмарка)        src/models/benchmarks.py
   ↓
⑤ Расчёт 5 блоков оценки           src/scoring/
   ├─ A. Business Quality
   ├─ B. Valuation
   ├─ C. Technical State
   ├─ D. Risk
   └─ E. Style Fit
   ↓
⑥ Горизонтальные оценки            src/engine/engine.py
   ├─ Short-term  (0–100)
   ├─ Medium-term (0–100)
   └─ Long-term   (0–100)
   ↓
⑦ Стоп-факторы                     src/engine/engine.py
   ↓
⑧ Итоговый рейтинг + решение       src/engine/engine.py · src/output/formatter.py
```

Ключевой принцип: **акция оценивается не в абсолюте, а относительно эталона своего класса**.  
Nvidia сравнивается с «идеальной Hypergrowth Tech», не с «идеальной Coca-Cola».

---

## 1. Шаг 1 — Получение сырых данных

**Файлы:** `src/data/price.py`, `src/data/sec_edgar.py`, `src/data/cache.py`

### 1.1 Ценовые данные (`price.py` → yfinance)

| Что берётся | Параметр / период | Влияние на результат |
|---|---|---|
| OHLCV (daily closes) | последние **2 года** (`period="2y"`) | Технический анализ, MA50/MA200, momentum, drawdown |
| `info` dict (yfinance) | snapshot | P/E trailing/forward, market cap, beta, sector, dividend yield |

> ⚙️ **Настраиваемо:** период загрузки цен — сейчас `"2y"`. Если нужны MA200 надёжно — минимум 1 год (~252 бара). Если хочется более длинный momentum — увеличить до `"3y"`.

### 1.2 Фундаментальные данные (`sec_edgar.py` → SEC EDGAR XBRL)

Берутся годовые отчёты (10-K). Система тянет до **10 лет истории**.

| Метрика | Тег XBRL |
|---|---|
| Revenue | `Revenues` / `RevenueFromContractWithCustomerExcludingAssessedTax` |
| Gross Profit | `GrossProfit` |
| Operating Income | `OperatingIncomeLoss` |
| Net Income | `NetIncomeLoss` |
| EPS Diluted | `EarningsPerShareDiluted` |
| Operating Cash Flow | `NetCashProvidedByUsedInOperatingActivities` |
| CapEx | `PaymentsToAcquirePropertyPlantAndEquipment` |
| Equity | `StockholdersEquity` |
| Total Assets | `Assets` |
| Total Liabilities | `Liabilities` |
| Long-term Debt | `LongTermDebt` |
| R&D Expense | `ResearchAndDevelopmentExpense` |

> ⚠️ **Важно для аналитика:** SEC EDGAR содержит только US-листинг (EDGAR CIK). Для иностранных компаний данные берутся только из `yfinance info` (неполные). Качество данных отражается в поле `data_quality` нормализатора (`"good"` / `"partial"` / `"poor"`).

### 1.3 Кеш (`cache.py`)

Все запросы кешируются на диск с TTL. Повторный запрос одного тикера в течение TTL не делает сетевых вызовов.

> ⚙️ **Настраиваемо:** TTL кеша. По умолчанию — несколько часов для цен и 24ч для фундаментала.

---

## 2. Шаг 2 — Нормализация данных

**Файл:** `src/data/normalizer.py` → функция `normalise()` → объект `NormalisedData`

### 2.1 Что происходит

1. Годовые ряды из EDGAR выравниваются по фискальным годам (oldest → newest).
2. Вычисляются **производные метрики**:

| Производная метрика | Формула |
|---|---|
| `fcf_annual` | `operating_cf − capex` |
| `gross_margin_annual` | `gross_profit / revenue × 100` |
| `operating_margin_annual` | `operating_income / revenue × 100` |
| `net_margin_annual` | `net_income / revenue × 100` |
| `revenue_growth_annual` | `(rev[t] − rev[t-1]) / |rev[t-1]| × 100` (YoY %) |
| `eps_growth_annual` | `(eps[t] − eps[t-1]) / |eps[t-1]| × 100` |
| `roe_annual` | `net_income / equity × 100` |
| `roa_annual` | `net_income / total_assets × 100` |
| `debt_to_equity_annual` | `long_term_debt / equity` |

3. Скалярные значения из yfinance (`pe_trailing`, `pe_forward`, `market_cap`, `beta`, `sector`, `dividend_yield`) кладутся напрямую.
4. Закрытия цен (`close_prices`) берутся из OHLCV DataFrame.

### 2.2 Оценка качества данных

После нормализации выставляется `data_quality`:

| Значение | Условие |
|---|---|
| `"good"` | ≥ 4 лет фундаментала, выручка и FCF доступны |
| `"partial"` | 2–3 года или частичные данные |
| `"poor"` | < 2 лет или критичные метрики отсутствуют |

> ✅ **Реализовано:** `data_quality` переносится в поле `data_confidence` объекта `AnalysisResult`. Форматтер выводит предупреждения: `🔴 DATA QUALITY: POOR` или `🟡 DATA QUALITY: PARTIAL` в заголовке отчёта и краткой строке. Score выдаётся в любом случае, но аналитик видит предупреждение.

### 2.3 Что НЕ делает нормализатор (точки роста)

- **Не сравнивает с медианой сектора** — хотя plan.md это упоминает, sector-relative нормализация не реализована. Сейчас оценивается абсолютное значение метрики.
- **Не корректирует разовые события** (write-offs, one-time charges) — искажения возможны.

---

## 3. Шаг 3 — Классификация типа компании

**Файл:** `src/classifier.py` → функция `classify(nd)` → `ClassificationResult`

### 3.1 Типы компаний

| Тип | Ключевые сигналы классификации |
|---|---|
| **Hypergrowth Tech** | revenue growth > 20%, gross margin > 60%, дивиденды слабые/нулевые, сектор Technology |
| **Mature Tech** | рост 5–15%, высокая маржа, стабильная прибыль, сектор Technology / Comm. Services |
| **Pharma / Healthcare** | R&D / revenue > 10%, сектор Healthcare / Biotechnology |
| **Dividend / Defensive** | dividend yield > 2%, стабильный FCF, низкий рост, сектор Utilities / Consumer Defensive |
| **Cyclical** | сектор Energy / Materials / Industrials / Consumer Cyclical, нестабильная прибыль |
| **Financial** | сектор Financials / Banks / Insurance |
| **Turnaround** | отрицательная прибыль + признаки восстановления (рост выручки, сокращение убытка) |
| **Other** | не попадает ни в одну категорию с достаточной уверенностью |

### 3.2 Алгоритм классификации (rule-based scoring)

Для каждого типа считается **score 0–100** по набору сигналов. Побеждает тип с наибольшим score. Если score победителя < 30% (порог `MIN_CONFIDENCE`) — присваивается `OTHER`.

**Сигналы и их логика:**

| Сигнал | Откуда | Что усиливает |
|---|---|---|
| `avg_revenue_growth` (3y) | `revenue_growth_annual` | Hypergrowth, Turnaround |
| `avg_gross_margin` (3y) | `gross_margin_annual` | Hypergrowth, Mature Tech, Pharma |
| `avg_net_margin` (3y) | `net_margin_annual` | Mature Tech, Dividend |
| `dividend_yield` | yfinance `info` | Dividend / Defensive |
| `rd_to_revenue` | `rd_expense / revenue` | Pharma |
| `debt_to_equity` (latest) | `debt_to_equity_annual` | Financial (терпимость к долгу) |
| `sector` (yfinance) | `nd.sector` | Sector hint → стартовый бонус |
| Убыточность | `net_income < 0` | Turnaround |
| FCF consistency | доля лет FCF > 0 | Dividend / Mature Tech |

**Sector hint** — yfinance sector добавляет стартовые очки нужному типу, но не переопределяет финансовые сигналы:

| Sector | Hint |
|---|---|
| Technology, Communication Services | +Mature Tech (далее уточняется через рост) |
| Healthcare, Biotechnology | +Pharma |
| Financials, Banks, Insurance | +Financial |
| Energy, Materials, Industrials, Consumer Cyclical | +Cyclical |
| Utilities, Real Estate, Consumer Defensive | +Dividend / Defensive |

**Защита от перекоса sector hint:**
- Hint `+8` (было `+15`) — финансовые сигналы имеют приоритет над сектором.
- Для Hypergrowth/Mature Tech hint выдаётся **только если** финансовый score > 0 (компания уже набрала очки без подсказки).
- Для Mature Tech hint **не выдаётся** если `avg_net_margin < −2%` (убыточная компания не может быть Mature Tech).
- Turnaround получает `+10` бонус если `avg_net_margin < −2% И` есть хотя бы 1 убыточный год.

> ⚙️ **Настраиваемо:**  
> — Порог `MIN_CONFIDENCE` (сейчас 0.30) — снижение даёт больше классифицированных акций, повышение — более строгую классификацию.  
> — Веса сигналов внутри каждого типа (жёстко заданы в `classify()`).  
> — Список sector hints в `_SECTOR_HINTS`.

> ⚠️ **Известные ограничения классификации:**  
> — Нет отдельного класса **«Mega-cap Quality Compounder»** (AAPL, MSFT, GOOGL) — они попадают в Mature Tech, хотя по профилю отличаются.  
> — Tech-компания с ростом 5% и низким убытком может ошибочно попасть в Turnaround вместо Mature Tech.

---

## 4. Шаг 4 — Выбор эталона (бенчмарка)

**Файл:** `src/models/benchmarks.py` → функция `get_benchmark(CompanyType)` → объект `Benchmark`

Эталон содержит два компонента:
1. **Веса блоков** — как блоки A–E складываются в итоговый Overall Score.
2. **Шкалы (Threshold)** — для каждой метрики: как сырое значение переводится в балл 0–10.

### 4.1 Веса блоков по типу компании

| Тип | Quality | Valuation | Technical | Risk | Style Fit |
|---|---|---|---|---|---|
| Hypergrowth Tech | **30%** | 20% | 20% | 15% | 15% |
| Mature Tech | **30%** | 25% | 15% | **20%** | 10% |
| Pharma | 28% | 22% | 15% | **20%** | 15% |
| Dividend / Defensive | 25% | 25% | 10% | **25%** | 15% |
| Cyclical | 25% | **28%** | 17% | 20% | 10% |
| Financial | 28% | **27%** | 13% | **22%** | 10% |
| Turnaround | 22% | **28%** | 20% | 18% | 12% |
| Other (fallback) | 25% | 25% | 20% | 20% | 10% |

> ⚙️ **Настраиваемо:** веса задаются в `BlockWeights` внутри каждой функции `_hypergrowth_tech()`, `_mature_tech()` и т.д. Изменение весов напрямую меняет итоговый Overall Score.

### 4.2 Шкалы (Threshold) — как работает оценка метрик

Каждая метрика имеет набор **опорных точек** `(raw_value → score)`. Между точками — **линейная интерполяция**. Выход за границы — clamp (прижать к крайнему значению).

**Принципиальный момент:** одна и та же метрика оценивается по-разному в зависимости от типа компании.

#### Пример: Revenue Growth

| Тип | Опорные точки шкалы |
|---|---|
| Hypergrowth Tech | −30%→0, 0%→2, **15%→5, 25%→7.5, 40%→10** |
| Mature Tech | −30%→0, 0%→2, **3%→5, 8%→7.5, 15%→10** |
| Dividend/Defensive | −30%→0, 0%→2, **−2%→5, 2%→7.5, 6%→10** |
| Turnaround | −30%→0, 0%→2, **−5%→5, 5%→7.5, 20%→10** |

Рост 10% YoY: для Hypergrowth Tech — балл ~5.7 (средне), для Mature Tech — ~7.9 (хорошо), для Dividend — максимум 10.

### 4.3 Полные шкалы по блокам (реализованные в коде)

#### Quality-метрики

| Метрика | Hypergrowth | Mature Tech | Pharma | Dividend | Cyclical | Financial | Turnaround |
|---|---|---|---|---|---|---|---|
| Revenue growth (%) | −30→0 / 15→5 / 40→10 | −30→0 / 3→5 / 15→10 | −30→0 / 2→5 / 15→10 | −30→0 / −2→5 / 6→10 | −30→0 / −5→5 / 15→10 | −30→0 / 0→5 / 12→10 | −30→0 / −5→5 / 20→10 |
| Gross margin (%) | 30→0 / 65→6 / 80→10 | 30→0 / 55→6 / 70→10 | 40→0 / 65→6 / 80→10 | 20→0 / 48→6 / 60→10 | 10→0 / 30→6 / 45→10 | н/д | 10→0 / 38→6 / 55→10 |
| Operating margin (%) | 0→0 / 20→6 / 35→10 | 5→0 / 22→6 / 30→10 | 5→0 / 22→6 / 32→10 | 8→0 / 20→6 / 28→10 | 3→0 / 14→6 / 22→10 | н/д | −10→0 / 8→6 / 18→10 |
| Net margin (%) | 0→0 / 15→6 / 25→10 | 5→0 / 18→6 / 25→10 | 5→0 / 18→6 / 28→10 | 5→0 / 15→6 / 22→10 | 2→0 / 10→6 / 16→10 | 10→0 / 25→6 / 35→10 | −15→0 / 6→6 / 14→10 |
| ROE (%) | 0→0 / 30→7 / 50→10 | 5→0 / 25→7 / 40→10 | 5→0 / 22→7 / 35→10 | 5→0 / 18→7 / 28→10 | 3→0 / 18→7 / 28→10 | 5→0 / 15→7 / 25→10 | −20→0 / 10→7 / 22→10 |
| FCF margin (%) | −5→0 / 15→7 / 25→10 | 0→0 / 18→7 / 28→10 | 0→0 / 16→7 / 25→10 | 3→0 / 14→7 / 20→10 | −5→0 / 10→7 / 18→10 | н/д | −10→0 / 8→7 / 18→10 |

#### Valuation-метрики (P/E trailing)

| Тип | Дёшево (→10) | Справедливо | Дорого | Очень дорого (→0) |
|---|---|---|---|---|
| Hypergrowth Tech | 10 | 30→8 | 60→5 / 100→2 | 150 |
| Mature Tech | 12 | 20→7 | 30→4 | 45 |
| Pharma | 12 | 18→7 | 28→4 | 45 |
| Dividend / Defensive | 10 | 16→7 | 24→4 | 35 |
| Cyclical | 5 | 10→7 | 18→4 | 30 |
| Financial | 6 | 10→7 | 16→4 | 25 |
| Turnaround | 5 | 12→7 | 22→4 | 40 |

> ⚙️ **Настраиваемо:** все опорные точки в `benchmarks.py`. Для настройки под рыночный режим (например, в период высоких ставок P/E должны быть ниже) — достаточно поправить шкалы P/E в нужных типах.

#### Risk-метрики (D/E)

| Тип | Safe (→10) | Moderate | High | Danger (→0) |
|---|---|---|---|---|
| Hypergrowth Tech | D/E 0.2 | 0.8→7 | 2.0→3 | 4.0 |
| Mature Tech | D/E 0.3 | 1.0→7 | 2.5→3 | 5.0 |
| Pharma | D/E 0.3 | 1.0→7 | 2.5→3 | 5.0 |
| Dividend / Defensive | D/E 0.3 | 1.0→7 | 2.0→3 | 4.0 |
| Cyclical | D/E 0.3 | 1.0→7 | 2.5→3 | 5.0 |
| Financial | **D/E не используется** — структурный леверидж нормален | — | — | — |
| Turnaround | D/E 0.5 | 1.5→7 | 3.5→3 | 7.0 |

> ✅ **Реализовано:** `score_risk(nd, bm, company_type)` принимает тип компании. Для `CompanyType.FINANCIAL` метрика `debt_to_equity` полностью пропускается (`_DE_EXEMPT_TYPES`). Стоп-фактор `High Debt` также имеет тип-специфичные пороги (см. §7).

#### Style Fit-метрики (тип-специфичные шкалы `_style`)

| Тип | Style-метрики |
|---|---|
| Hypergrowth Tech | `revenue_growth_style` (20→5 / 45→10), `gross_margin_style` (55→6 / 85→10) |
| Mature Tech | `revenue_growth_style` (4→5 / 16→10), `fcf_margin_style` (15→7 / 30→10) |
| Pharma | `rd_to_revenue` (5→3 / 20→10 / 40→8 — есть оптимум!), `gross_margin_style` (65→6 / 85→10) |
| Dividend / Defensive | `dividend_yield_pct` (0→2 / 2.5→7 / 4.5→10 / 9→5 — штраф за слишком высокий yield) |
| Cyclical | `revenue_growth_style` (0→5 / 18→10) |
| Financial | `revenue_growth_style` (2→5 / 12→10) |
| Turnaround | `revenue_growth_style` (0→5 / 25→10) |

> ⚙️ **Настраиваемо:** добавить новые `_style`-метрики в benchmark + реализовать extractor в `style_fit.py → _METRIC_EXTRACTORS`.

---

## 5. Шаг 5 — Пять блоков оценки

**Файлы:** `src/scoring/quality.py`, `valuation.py`, `technical.py`, `risk.py`, `style_fit.py`

Каждый блок возвращает объект `BlockScore` со score **0–10**, детальным разбивкой (`breakdown`) по метрикам и текстовыми заметками (`notes`) для аналитика.

**Агрегация внутри блока:** простое среднее по доступным метрикам. Если метрика недоступна (NaN) — она **пропускается**, не тянет результат вниз.

---

### Блок A — Business Quality

**Файл:** `src/scoring/quality.py`  
*Вопрос: насколько бизнес фундаментально сильный?*

| Метрика | Как берётся | Окно |
|---|---|---|
| Revenue Growth | среднее `revenue_growth_annual` | последние 3 года |
| EPS Growth | среднее `eps_growth_annual` | последние 3 года |
| Gross Margin | среднее `gross_margin_annual` | последние 3 года |
| Operating Margin | среднее `operating_margin_annual` | последние 3 года |
| Net Margin | среднее `net_margin_annual` | последние 3 года |
| ROE | среднее `roe_annual` | последние 3 года |
| FCF Margin | `FCF / Revenue × 100` — среднее за 3 года | последние 3 года |

Каждая метрика оценивается через шкалу бенчмарка → балл 0–10. Финальный балл блока = среднее.

**Автоматические заметки (analyst notes):**

| Условие | Текст заметки |
|---|---|
| Revenue growth > 20% | `"strong revenue growth X%"` |
| Revenue growth < 0% | `"declining revenue X%"` |
| Gross margin > 60% | `"high gross margin X%"` |
| ROE > 30% | `"strong ROE X%"` |
| FCF margin > 15% | `"strong FCF margin X%"` |

> ⚙️ **Настраиваемо:** пороги заметок — в `quality.py`. Окно усреднения `n=3` — в функции `_recent_mean()`.

> ⚠️ **Ограничение:** EPS growth нестабилен при смене прибыльности — CV может быть очень высоким. Лучше дополнить нормализованным EPS (adjusted).

---

### Блок B — Valuation

**Файл:** `src/scoring/valuation.py`  
*Вопрос: дорого или дёшево стоит акция?*

| Метрика | Формула | Источник |
|---|---|---|
| P/E Trailing | из yfinance | `nd.pe_trailing` |
| P/E Forward | из yfinance | `nd.pe_forward` |
| P/S | `market_cap / revenue_latest` | вычисляется внутри |
| FCF Yield | `fcf_latest / market_cap × 100` | вычисляется внутри |
| PEG Ratio | `pe_forward / eps_growth_avg_3y` | вычисляется внутри (только при росте > 0) |
| EV/EBITDA | `(market_cap + LTD − cash) / ebitda_latest` | вычисляется внутри; EBITDA = `operating_income + D&A` |

**Логика направления:** P/E, P/S и PEG — чем ниже, тем выше балл. FCF Yield — чем выше, тем лучше (обратная зависимость).

**P/S шкала — тип-специфична (из бенчмарка):**

| Тип | Дёшево (→10) | Справедливо (→7) | Дорого (→3) | Очень дорого (→0) |
|---|---|---|---|---|
| Hypergrowth Tech | P/S 2 | 6 | 15 | 30 |
| Mature Tech | P/S 1 | 3 | 8 | 15 |
| Pharma | P/S 1 | 3 | 8 | 15 |
| Dividend / Defensive | P/S 0.5 | 1.5 | 4 | 8 |
| Cyclical | P/S 0.3 | 1 | 2.5 | 5 |
| Financial | P/S 0.5 | 1.5 | 4 | 8 |
| Turnaround | P/S 0.3 | 1.5 | 5 | 10 |
| Other | P/S 0.5 | 2 | 6 | 12 |

**FCF Yield шкала — тип-специфична (из бенчмарка):**

| Тип | Слабо (→0) | Норма (→4) | Хорошо (→7) | Отлично (→10) |
|---|---|---|---|---|
| Hypergrowth Tech | 0% | 1% | 2% | 7% |
| Mature Tech | 0% | 1.5% | 3% | 10% |
| Pharma | 0% | 1% | 3% | 8% |
| Dividend / Defensive | 0% | 2% | 4% | 12% |
| Cyclical | 0% | 1% | 3% | 8% |
| Financial | 0% | 1.5% | 3.5% | 9% |
| Turnaround | 0% | 0.5% | 2% | 6% |

**PEG Ratio шкала — только для growth-типов:**

| Тип | Отлично (→10) | Хорошо (→7) | Справедливо (→4) | Дорого (→0) |
|---|---|---|---|---|
| Hypergrowth Tech | PEG 0.5 | 1.0 | 2.0 | 4.0 |
| Mature Tech | PEG 0.8 | 1.5 | 2.5 | 4.0 |
| Pharma | PEG 0.8 | 1.5 | 2.5 | 4.0 |
| Turnaround | PEG 0.5 | 1.5 | 3.0 | 6.0 |

> PEG **не рассчитывается** для Dividend, Cyclical, Financial (рост не является тезисом этих типов), а также при `eps_growth ≤ 0`.

**EV/EBITDA шкала — тип-специфична:**

| Тип | Дёшево (→10) | Справедливо (→7) | Дорого (→3) | Очень дорого (→0) |
|---|---|---|---|---|
| Hypergrowth Tech | ×20 | 40 | 80 | 150 |
| Mature Tech | ×10 | 18 | 30 | 50 |
| Pharma | ×10 | 18 | 30 | 50 |
| Dividend / Defensive | ×6 | 12 | 20 | 35 |
| Cyclical | ×4 | 8 | 15 | 25 |
| Financial | не используется | — | — | — |
| Turnaround | ×5 | 12 | 22 | 40 |
| Other | ×8 | 15 | 28 | 50 |

> EV/EBITDA **пропускается** если EBITDA ≤ 0 (убыточная компания). D&A берётся из SEC EDGAR (`DepreciationDepletionAndAmortization`).

**Автоматические заметки:**

| Условие | Текст |
|---|---|
| P/E trailing > 60 | `"P/E X — very expensive"` |
| P/E trailing < 12 | `"P/E X — cheap"` |
| P/S > порога бенчмарка | `"P/S X — elevated"` |
| P/S < порога бенчмарка | `"P/S X — attractive"` |
| FCF yield > 5% | `"FCF yield X% — attractive"` |
| FCF yield < 0% | `"negative FCF yield"` |
| PEG < 1.0 | `"PEG X — attractive (growth underpriced)"` |
| PEG > 3.0 | `"PEG X — expensive relative to growth"` |
| EV/EBITDA < 8 | `"EV/EBITDA X — cheap"` |
| EV/EBITDA > 40 | `"EV/EBITDA X — very expensive"` |

> ✅ **Реализовано:** P/S, FCF yield, PEG, EV/EBITDA — все в `benchmarks.py` (тип-специфичные). EBITDA и D&A предвычисляются в `normalizer.py`.

> ⚙️ **Настраиваемо:** все шкалы в `benchmarks.py`.

---

### Блок C — Technical State

**Файл:** `src/scoring/technical.py`  
*Вопрос: как ведёт себя акция на рынке прямо сейчас?*

Источник данных: `nd.close_prices` — последние до 252 торговых дней.  
**Блок не зависит от бенчмарка** — технические сигналы универсальны для всех типов.

| Метрика | Расчёт | Минимум баров | Шкала |
|---|---|---|---|
| Price vs MA50 | `(price / MA50 − 1) × 100%` | 50 | −20%→0 / 0%→7 / 5%→9 / 40%→7 (extended штраф) |
| Price vs MA200 | `(price / MA200 − 1) × 100%` | 200 | аналогично MA50 |
| Momentum 3m | % изменение за 63 дня | 64 | −30%→0 / 0%→4 / 25%→8 / 50%→10 |
| Momentum 6m | % изменение за 126 дней | 127 | аналогично 3m |
| Momentum 12m | % изменение за 252 дня | 253 | аналогично 3m |
| Drawdown от максимума | `(price − peak_52w) / peak_52w × 100%` | 1 | 0%→10 / −10%→8 / −25%→5 / −50%→1 |
| Trend Quality | % дней выше MA50 за последние 63 дня | 10 | 0%→0 / 50%→5 / 85%→10 |
| **Relative Strength vs SPY** | `momentum_3m(stock) − momentum_3m(SPY)` | 64 + SPY данные | −30%→0 / −15%→2 / 0%→6 / +3%→8 / +10%→10 |

**Особые случаи:**
- Менее 50 баров → возвращается нейтральный балл **5.0** (нет данных, нет суждения)
- Менее 200 баров → MA200 и momentum 6m пропускаются
- Менее 253 баров → momentum 12m пропускается
- Менее 64 баров → momentum 3m пропускается
- `spy_close_prices` не заполнены → Relative Strength пропускается

**Автоматические заметки:**

| Условие | Текст |
|---|---|
| Price > MA200 | `"price above MA200"` |
| Price < MA200 | `"below MA200"` |
| Drawdown < −30% | `"deep drawdown X%"` |
| Trend quality > 75% | `"strong trend quality"` |
| RS > +10% vs SPY (3m) | `"outperforming SPY by X% (3m)"` |
| RS < −10% vs SPY (3m) | `"underperforming SPY by X% (3m)"` |

> ⚙️ **Настраиваемо:** все шкалы в `technical.py`. SPY-цены фетчатся в `analyse()` параллельно с акцией и передаются через `nd.spy_close_prices`.

> ✅ **Реализовано:** Relative Strength vs SPY, Momentum 12m. Volatility regime — пока не реализован.

---

### Блок D — Risk

**Файл:** `src/scoring/risk.py`  
*Вопрос: насколько рискованно вкладываться? (10 = минимальный риск)*

| Метрика | Расчёт | Что оценивает |
|---|---|---|
| Debt-to-Equity | `long_term_debt / equity` (latest) | долговая нагрузка |
| Beta | из yfinance | рыночная волатильность |
| Earnings Stability | `std / max(abs_mean, median_abs, 1.0)` net margin за все годы | стабильность прибыли (робастный к нулю) |
| FCF Consistency | доля лет с FCF > 0 | надёжность денежного потока |
| Revenue Stability | CV (`std / mean`) revenue growth за все годы | предсказуемость роста |
| Dilution Risk | средний YoY рост `shares_outstanding` из EDGAR (%) | размывание акционеров |

**Шкалы:**

| Метрика | Низкий риск (→10) | Средний | Высокий (→0) |
|---|---|---|---|
| Earnings Stability (CV) | CV 0 → 10 | CV 0.7 → 5 | CV 2.0 → 0 |
| FCF Consistency | 100% лет → 10 | 75% → 7 | 0% → 0 |
| Revenue Stability (CV) | CV 0 → 10 | CV 1.0 → 4 | CV 3.0 → 0 |
| Dilution Risk | ≤−5%/год (байбэк) → 10 | 0% → 8 / 2% → 6 | 10%+/год → 0 |

Beta шкала — тип-специфична (из бенчмарка). D/E шкала — тип-специфична (см. §4.3).

**Автоматические заметки:**

| Условие | Текст |
|---|---|
| D/E > 3.0 | `"high D/E X — elevated leverage risk"` |
| D/E < 0.3 | `"low D/E X — strong balance sheet"` |
| Beta > 2.0 | `"beta X — high volatility"` |
| Beta < 0.7 | `"beta X — low market sensitivity"` |
| Earnings instability > 0.8 | `"earnings instability (score=X)"` |
| FCF positive < 50% лет | `"FCF negative in majority of years"` |
| Revenue growth CV > 1.5 | `"highly volatile revenue growth (CV=X)"` |
| Dilution среднее > 5%/год | `"significant share dilution (X%/yr avg)"` |
| Байбэк среднее < −1%/год | `"share buybacks (X%/yr avg — positive)"` |

**Fallback:** если ни одна метрика недоступна — балл 5.0 (нейтральный) + заметка `"insufficient data"`.

> ⚙️ **Настраиваемо:** пороги заметок в `risk.py`. Шкалы D/E и Beta — в `benchmarks.py` (тип-специфично).

> ✅ **Реализовано:** `score_risk()` принимает `company_type`. D/E пропускается для `FINANCIAL`. Earnings Stability использует робастную формулу. Dilution Risk: `shares_outstanding_annual` из EDGAR (`CommonStockSharesOutstanding`), YoY% вычисляется в `normalizer.py` как `shares_dilution_annual`.

---

### Блок E — Style Fit

**Файл:** `src/scoring/style_fit.py`  
*Вопрос: насколько акция соответствует своему типу-архетипу?*

Блок использует только `_style`-метрики из бенчмарка — шкалы, откалиброванные под «идеального представителя» данного типа (более строгие, чем quality-шкалы).

**Реализованные extractors:**

| Ключ метрики | Формула | Типы, где используется |
|---|---|---|
| `revenue_growth_style` | среднее `revenue_growth_annual` за 3y | Hypergrowth, Mature, Cyclical, Financial, Turnaround |
| `gross_margin_style` | среднее `gross_margin_annual` за 3y | Hypergrowth, Pharma |
| `fcf_margin_style` | среднее `FCF/Revenue × 100` за 3y | Mature Tech |
| `rd_to_revenue` | `rd_expense / revenue × 100` (latest) | Pharma |
| `dividend_yield_pct` | `nd.dividend_yield × 100` | Dividend / Defensive |

**Fallback:** если у типа нет ни одной `_style`-метрики в бенчмарке → балл **5.0** + заметка `"no style-specific thresholds"`.

> ⚙️ **Как добавить новую style-метрику:**  
> 1. Добавить ключ `"metric_name_style"` в `thresholds` нужного бенчмарка в `benchmarks.py`  
> 2. Добавить extractor в `_METRIC_EXTRACTORS` в `style_fit.py`

---

## 6. Шаг 6 — Горизонтальные оценки

**Файл:** `src/engine/engine.py`

Одна и та же акция получает **три независимых score** — потому что разные горизонты требуют разных приоритетов блоков.

### Веса блоков по горизонту

| Блок | Short-term | Medium-term | Long-term |
|---|---|---|---|
| Quality | 10% | 25% | **35%** |
| Valuation | 20% | **30%** | 15% |
| Technical | **40%** | 20% | 10% |
| Risk | 25% | 15% | 25% |
| Style Fit | 5% | 10% | 15% |

**Логика приоритетов:**

- **Short-term (недели – 3 месяца):** доминирует технический анализ (40%) — тренд, моментум, точка входа. Risk на входе важен.
- **Medium-term (3–12 месяцев):** valuation (30%) — насколько есть апсайд. Quality начинает влиять.
- **Long-term (1–5 лет):** quality (35%) — фундаментальная сила бизнеса. Risk снова важен — на длинном горизонте слабый баланс убивает идею.

**Формула:** `score = weighted_avg(blocks) × 10` → диапазон **0–100**.

> ⚙️ **Настраиваемо:** веса в `_HORIZON_WEIGHTS` в `engine.py`. Изменение весов **не влияет** на Overall Score (он использует веса бенчмарка).

> ⚠️ **Не реализовано:** event-driven катализаторы (earnings beats, buybacks, M&A-ожидания), sector rotation как сигнал для medium-term.

---

## 7. Шаг 7 — Стоп-факторы

**Файл:** `src/engine/engine.py → _check_stop_factors()`

Стоп-факторы проверяются **независимо от score** — это структурные красные флаги, которые могут принудительно изменить итоговое решение.

### Реализованные стоп-факторы

| Название | Условие срабатывания | Severity | Эффект |
|---|---|---|---|
| **Negative FCF** | FCF < 0 в последних 2–3 годах | `warning` | Понижает доверие |
| **Negative FCF + High Debt** | FCF < 0 И D/E > 2.0 | **`critical`** | Решение → **Avoid** |
| **Extreme Valuation** | P/E trailing > 100 | `warning` | Понижает доверие |
| **High Debt** | D/E > порога (тип-специфично, см. ниже) | **`critical`** | Решение → **Avoid** |
| **Deteriorating Margins** | Net margin падает 3 года подряд И последнее значение < −10% | `warning` | Понижает доверие |
| **Technical Breakdown** | Technical Score < 3.0 / 10 | `warning` | Понижает доверие |
| **Low Liquidity** | `avg_volume` < 100 000 акций/день | `warning` | Очень неликвидный, широкий спред |
| **Limited Liquidity** | `avg_volume` 100 000–500 000 | `warning` | Ограниченная торгуемость |

**Правило:** при наличии хотя бы одного `critical` → итоговое решение принудительно становится **«Avoid»**, независимо от score.

### Тип-специфичные пороги High Debt

| Тип компании | Порог D/E | Логика |
|---|---|---|
| Financial | **не применяется** (`_DE_STOP_EXEMPT`) | леверидж структурный для банков |
| Turnaround | D/E > **8.0** | реструктуризация допускает временный долг |
| Cyclical | D/E > **6.0** | цикличные бизнесы капиталоёмки |
| Остальные | D/E > **4.0** | стандартный порог |

> ⚙️ **Настраиваемо:** пороги в `_DE_CRITICAL_THRESHOLD` и `_DE_STOP_EXEMPT` в `engine.py`. Для добавления нового стоп-фактора — добавить `StopFactor(...)` в `_check_stop_factors()`.

---

## 8. Шаг 8 — Итоговый рейтинг и решение

**Файл:** `src/engine/engine.py`

### Overall Score

Рассчитывается как взвешенная сумма 5 блоков по весам **бенчмарка** (не горизонтальным весам):

```
overall = weighted_avg(block_scores, benchmark.weights) × 10   →  0–100
```

### Рейтинг (Rating)

| Overall Score | Рейтинг |
|---|---|
| 85–100 | Strong Candidate |
| 70–84 | Good Candidate |
| 55–69 | Neutral / Watchlist |
| 40–54 | Weak |
| < 40 | Avoid |

### Финальное решение (Decision)

| Overall Score | Стоп-факторы | Решение |
|---|---|---|
| ≥ 70 | нет `critical` | **Buy** |
| 55–69 | нет `critical` | **Watch** |
| 40–54 | нет `critical` | **Hold** |
| < 40 | любые | **Avoid** |
| любой | есть `critical` | **Avoid** |

> ✅ **Выровнено:** порог Buy (≥ 70) совпадает с рейтингом "Good Candidate" (≥ 70) — больше нет зазора, когда акция "Good Candidate" но решение "Watch".

> ⚙️ **Настраиваемо:** пороги рейтинга — в `_rating()`, пороги решения — в `_decision()` в `engine.py`.

---

## 9. Вывод результата

**Файл:** `src/output/formatter.py` — функции `format_report()` и `format_brief()`

Итоговый отчёт включает:
- Тикер, тип компании, уверенность классификации
- **Предупреждение о качестве данных** (если `data_confidence ≠ "good"`):
  - `🔴 DATA QUALITY: POOR` — в заголовке отчёта и краткой строке
  - `🟡 DATA QUALITY: PARTIAL` — аналогично
- Баллы по 5 блокам с детальным breakdown и заметками
- Три горизонтальных score (short / medium / long)
- Overall Score + Rating + Decision
- Список сработавших стоп-факторов с severity

---

## 10. Открытые вопросы для аналитика

### По классификации

- Нужен ли класс **«Mega-cap Quality Compounder»** для AAPL, MSFT, GOOGL? Сейчас они падают в Mature Tech, хотя профиль существенно другой.
- Как правильно классифицировать tech-компанию с ростом 5–8% и высокой маржой — Mature Tech или отдельный класс «Quality Compounder»?
- Как обрабатывать иностранные ADR (не в EDGAR) — данные частичные, классификация может быть неверной.

### По шкалам и метрикам

- Для **Pharma** — R&D/revenue как style-метрика разумна, но не учитывает стадию pipeline. Нужна ли качественная оценка (Phase II/III)?
- Для **Cyclical** — нужна ли поправка на фазу commodity-цикла? Сейчас шкалы статичны.
- ~~Стоит ли добавить **PEG ratio**~~ — ✅ **Реализован** для Hypergrowth, Mature Tech, Pharma, Turnaround.
- ~~EV/EBITDA не реализован — критично для Cyclical и Financial анализа.~~ — ✅ **Реализован**: `ebitda_annual = operating_income + D&A` (из EDGAR), тип-специфичные шкалы во всех бенчмарках. Financial не использует EV/EBITDA.
- **Sector-relative нормализация** (сравнение с медианой сектора) из плана не реализована — нужно ли добавить?

### По стоп-факторам

- ~~Порог `High Debt` (D/E > 4.0) не работает корректно для Financial-сектора~~ — ✅ **Исправлено**: `FINANCIAL` исключён из стоп-фактора (`_DE_STOP_EXEMPT`), также пропускается D/E в Risk-блоке. Для банков можно добавить отдельный индикатор CET1 ratio.
- ~~Нужен ли стоп **«слишком маленький free float» или **«низкая ликвидность»** (avg volume)?~~ — ✅ **Реализовано**: `avg_volume` из yfinance (`averageVolume`). Срабатывают `Low Liquidity` (< 100k) и `Limited Liquidity` (< 500k) — оба `warning`.
- ~~Нужен ли стоп для **dilution risk** — агрессивная эмиссия акций у убыточных компаний?~~ — ✅ **Реализовано**: `dilution_risk` в Risk-блоке (байбэк → 10, >5%/год → 0).

### По горизонтам и решению

- Как учесть **event-driven катализаторы** (upcoming earnings, buyback announce, FDA decision)?
- Стоит ли добавить **sector rotation** как сигнал для medium-term score?
- Есть ли смысл в **4-м горизонте** — «very short term» (swing trade, 1–2 недели)?
- Должны ли горизонтальные веса тоже быть **тип-специфичны**? Для Dividend-акций техника менее важна на любом горизонте.

### По данным

- Как обрабатывать компании, только что вышедшие на IPO (< 2 лет истории)? `data_quality = "poor"` — score выдаётся, но форматтер показывает `🔴 DATA QUALITY: POOR`. ✅ Предупреждение реализовано; полное снижение score при poor data — открытый вопрос.
- Нужна ли поддержка **quarterly данных** (10-Q) помимо годовых (10-K) для более свежей оценки?
