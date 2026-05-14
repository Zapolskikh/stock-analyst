# Stock Analyst — High-Level Design

> Документ для разработчиков.  
> Описывает архитектуру модулей, их интерфейсы и взаимодействие.

---

## 1. Структура проекта

```
stock-analyst/
├── main.py                     # CLI entry point
├── fetch.py                    # CLI: скачать данные + чарты
├── pyproject.toml              # uv / PEP 621
│
├── src/
│   ├── data/
│   │   ├── cache.py            # TTL file cache (parquet / json)
│   │   ├── price.py            # yfinance wrapper
│   │   ├── sec_edgar.py        # SEC EDGAR XBRL fetcher
│   │   └── normalizer.py       # raw → NormalisedData
│   │
│   ├── classifier.py           # NormalisedData → CompanyType
│   │
│   ├── models/
│   │   └── benchmarks.py       # CompanyType → Benchmark (weights + thresholds)
│   │
│   ├── scoring/
│   │   ├── base.py             # BlockScore dataclass + avg_scores()
│   │   ├── quality.py          # score_quality(nd, bm) → BlockScore
│   │   ├── valuation.py        # score_valuation(nd, bm) → BlockScore
│   │   ├── technical.py        # score_technical(nd) → BlockScore
│   │   ├── risk.py             # score_risk(nd, bm) → BlockScore
│   │   └── style_fit.py        # score_style_fit(nd, bm) → BlockScore
│   │
│   ├── engine/
│   │   └── engine.py           # analyse(ticker) / analyse_nd(nd) → AnalysisResult
│   │
│   ├── output/
│   │   └── formatter.py        # format_report() / format_brief()
│   │
│   └── charts/
│       ├── price_chart.py      # OHLCV + MA + RSI + MACD (Plotly)
│       └── fundamental_chart.py # 6 fundamental charts
│
└── tests/
    ├── test_cache.py           # 8 tests
    ├── test_price.py           # 8 tests
    ├── test_sec_edgar.py       # 12 tests
    ├── test_charts.py          # 10 tests
    ├── test_normalizer.py      # 48 tests
    ├── test_classifier.py      # 36 tests
    ├── test_benchmarks.py      # 68 tests
    ├── test_scoring.py         # 52 tests
    └── test_engine.py          # 52 tests  — всего: 294
```

---

## 2. Data Flow

```
ticker (str)
    │
    ▼
[src/data/price.py]
  fetch_ohlcv()    →  DataFrame (OHLCV, 2 года)
  fetch_info()     →  dict (yfinance .info)
    │
[src/data/sec_edgar.py]
  fetch_fundamentals()  →  dict[str, DataFrame]
    │                       (каждый ключ = XBRL concept, строки = периоды)
    ▼
[src/data/normalizer.py]
  normalise(fundamentals, price_df, info, ticker)
    →  NormalisedData
         ├── годовые ряды (revenue, EPS, FCF, margins, ROE, ...)
         ├── производные (growth rates, D/E, margins%)
         ├── scalars (current_price, pe, beta, sector, ...)
         ├── close_prices (последние 252 дня)
         └── data_quality: "good" | "partial" | "poor"
    │
    ▼
[src/classifier.py]
  classify(nd)  →  ClassificationResult
                    ├── company_type: CompanyType (enum, 8 значений)
                    ├── confidence: float 0–1
                    └── scores / signals (dict, для отладки)
    │
    ▼
[src/models/benchmarks.py]
  get_benchmark(company_type)  →  Benchmark
                                    ├── weights: BlockWeights
                                    └── thresholds: dict[str, Threshold]
    │
    ▼
[src/scoring/*]
  score_quality(nd, bm)    →  BlockScore
  score_valuation(nd, bm)  →  BlockScore
  score_technical(nd)      →  BlockScore   (без bm — только цены)
  score_risk(nd, bm, company_type)  →  BlockScore   (company_type влияет на D/E)
  score_style_fit(nd, bm)  →  BlockScore
    │
    ▼
[src/engine/engine.py]
  analyse_nd(nd)  →  AnalysisResult
                      ├── block_scores: dict[str, BlockScore]
                      ├── horizon: HorizonScores (short / medium / long)
                      ├── overall_score: float 0–100
                      ├── stop_factors: list[StopFactor]
                      ├── rating: str
                      └── decision: str
    │
    ▼
[src/output/formatter.py]
  format_report(result)  →  str (полный отчёт)
  format_brief(result)   →  str (одна строка для таблицы)
```

---

## 3. Ключевые типы данных

### `NormalisedData` (dataclass)
```python
@dataclass
class NormalisedData:
    ticker: str
    years: list[int]                      # [2020, 2021, 2022, 2023]

    # Годовые ряды (raw, USD)
    revenue_annual: list[float]
    gross_profit_annual: list[float]
    net_income_annual: list[float]
    operating_cf_annual: list[float]
    capex_annual: list[float]
    equity_annual: list[float]
    long_term_debt_annual: list[float]
    eps_diluted_annual: list[float]
    rd_expense_annual: list[float]

    # Производные ряды (%)
    fcf_annual: list[float]
    gross_margin_annual: list[float]
    operating_margin_annual: list[float]
    net_margin_annual: list[float]
    revenue_growth_annual: list[float]    # YoY %, первый элемент = NaN
    eps_growth_annual: list[float]
    roe_annual: list[float]
    roa_annual: list[float]
    debt_to_equity_annual: list[float]

    # Скаляры из yfinance
    current_price: float | None
    market_cap: float | None
    pe_trailing: float | None
    pe_forward: float | None
    beta: float | None
    sector: str | None
    industry: str | None
    dividend_yield: float | None          # 0.0–1.0

    # Технический анализ
    close_prices: list[float]             # последние ≤ 252 дня

    # Качество данных
    years_of_history: int
    data_quality: str                     # "good" | "partial" | "poor"
    missing_metrics: list[str]
```

### `BlockScore` (dataclass)
```python
@dataclass
class BlockScore:
    score: float          # 0–10, автоматически clamped, NaN → 0
    breakdown: dict       # метрика → её балл (для отладки)
    notes: list[str]      # текстовые сигналы для отчёта
```

### `Benchmark` (dataclass)
```python
@dataclass
class Benchmark:
    company_type: CompanyType
    weights: BlockWeights         # quality, valuation, technical, risk, style_fit
    thresholds: dict[str, Threshold]

@dataclass
class Threshold:
    points: list[tuple[float, float]]   # [(raw_value, score), ...]
    # Линейная интерполяция между точками; clamp на краях
```

### `AnalysisResult` (dataclass)
```python
@dataclass
class AnalysisResult:
    ticker: str
    company_type: CompanyType
    classification_confidence: float
    block_scores: dict[str, BlockScore]
    horizon: HorizonScores        # .short / .medium / .long  (0–100)
    overall_score: float          # 0–100
    stop_factors: list[StopFactor]
    rating: str
    decision: str                 # "Buy" | "Watch" | "Hold" | "Avoid"
    data_confidence: str          # "good" | "partial" | "poor" — из NormalisedData.data_quality
```

---

## 4. Модули — детали

### `src/data/cache.py`
- **TTL file cache** на диске (`~/.stock_analyst_cache/`)
- Форматы: Parquet (для DataFrame), JSON (для dict)
- Ключ = `f"{ticker}_{dataset}"`, TTL по умолчанию 24 часа
- `get(key, ttl)` / `set(key, value)` / `invalidate(key)`

### `src/data/price.py`
- Тонкая обёртка над `yfinance`
- `fetch_ohlcv(ticker, period)` → DataFrame с колонками Open/High/Low/Close/Volume
- `fetch_info(ticker)` → dict (P/E, market cap, beta, sector, ...)
- `fetch_dividends(ticker)` / `fetch_splits(ticker)`
- Все вызовы проходят через cache

### `src/data/sec_edgar.py`
- Запросы к `data.sec.gov/api/xbrl/companyfacts/`
- `fetch_fundamentals(ticker)` → dict: ключ = XBRL concept, значение = DataFrame
- Поддерживаемые концепты: Revenues, NetIncomeLoss, EarningsPerShareDiluted,
  NetCashProvidedByUsedInOperatingActivities, PaymentsToAcquirePropertyPlantAndEquipment,
  StockholdersEquity, Assets, Liabilities, LongTermDebt, GrossProfit,
  ResearchAndDevelopmentExpense, OperatingIncomeLoss
- Фильтрует только годовые (10-K) периоды; дедуплицирует по fiscal year

### `src/data/normalizer.py`
- `normalise(fundamentals, price_df, info, ticker)` → `NormalisedData`
- Выравнивает все ряды по одному набору fiscal years
- Вычисляет FCF = operating_cf − capex
- Вычисляет все margin%, growth%, ROE, D/E
- Определяет `data_quality`: good (≥ 4 года, ≤ 2 пропуска), partial, poor

### `src/classifier.py`
- `classify(nd)` → `ClassificationResult`
- Каждый тип компании получает score по своим правилам (сигналы + пороги)
- Победитель = тип с наибольшим score
- Confidence = score_winner / sum(all_scores)
- Fallback: если ничего не набрало > 0 → OTHER

### `src/models/benchmarks.py`
- `get_benchmark(CompanyType)` → `Benchmark`
- 7 бенчмарков захардкожены; Others наследует от Mature Tech
- `Threshold.score(value)` — главный метод, линейная интерполяция

### `src/scoring/risk.py`
- `score_risk(nd, bm, company_type=CompanyType.OTHER)` → BlockScore
- `company_type` управляет исключениями: `CompanyType.FINANCIAL` — D/E не считается (`_DE_EXEMPT_TYPES`)
- Earnings stability: робастная формула `std / max(abs_mean, median_abs, 1.0)` — не взрывается при mean ≈ 0 (Turnaround)

### `src/scoring/technical.py`
- Единственный блок без `Benchmark` (нет зависимости от типа компании)
- Вспомогательные функции: `_sma(prices, n)`, `_momentum(prices, n)`,
  `_drawdown_from_high(prices)`, `_trend_quality(prices, window)`
- Порог нейтральности: < 50 точек цен → score = 5.0

### `src/engine/engine.py`
- `analyse(ticker)` — полный пайплайн с сетевыми запросами
- `analyse_nd(nd)` — офлайн вход (для тестов и батч-обработки)
- Горизонтальные веса задаются константой `_HORIZON_WEIGHTS` (dict)
- `_check_stop_factors(nd, blocks, company_type)` → list[StopFactor]
  - `_DE_STOP_EXEMPT = {CompanyType.FINANCIAL}` — High Debt стоп не применяется
  - `_DE_CRITICAL_THRESHOLD` — тип-специфичные пороги: Turnaround 8.0, Cyclical 6.0, остальные 4.0
- `_decision(score, stop_factors)` — critical стоп-факторы всегда возвращают «Avoid»
- `AnalysisResult.data_confidence` — заполняется из `nd.data_quality`

---

## 5. Тест-покрытие

| Модуль | Тестов | Стратегия |
|---|---|---|
| cache | 8 | TTL, форматы, инвалидация |
| price | 8 | mock yfinance |
| sec_edgar | 12 | mock requests |
| charts | 10 | smoke tests, no rendering |
| normalizer | 48 | edge cases: пустые ряды, NaN, частичные данные |
| classifier | 36 | каждый тип компании + граничные случаи |
| benchmarks | 68 | каждый бенчмарк + Threshold interpolation |
| scoring | 61 | каждый блок + helpers (включая PEG, type-specific P/S/FCF) |
| engine | 65 | pipeline, stop factors, formatter, data_confidence |
| **Итого** | **316** | |

Тесты полностью офлайн — никаких сетевых запросов.  
Запуск: `python -m pytest tests/ --tb=short`

---

## 6. Зависимости

| Пакет | Назначение |
|---|---|
| yfinance | Котировки, info, дивиденды |
| pandas | DataFrame операции |
| numpy | Математика |
| plotly | Интерактивные чарты |
| pyarrow | Parquet cache |
| requests | SEC EDGAR API |
| beautifulsoup4 | Парсинг (резерв) |

Runtime Python: **3.13**, менеджер: **uv**.

---

## 7. Точки расширения

| Область | Текущее | Возможное улучшение |
|---|---|---|
| Классификация | Rule-based scoring | ML classifier (features → type) |
| Веса бенчмарков | Захардкожены | Backtested / ML-оптимизированные |
| Технический анализ | MA, momentum 3m/6m/12m, drawdown, RS vs SPY | Sector rotation, volume profile, volatility regime |
| Стоп-факторы | 5 правил, type-aware D/E | Расширить: insider selling, short interest, analyst cuts |
| Источники данных | yfinance + SEC EDGAR | FRED (macro), earnings calendars, options flow |
| Вывод | Текстовый отчёт + data_confidence | HTML report, PDF, Telegram bot |
| Батч-обработка | Последовательно | Async / concurrent |
| Оценка оценки | EV/EBITDA: `ebitda_annual = oi + D&A` (из EDGAR), тип-спец. шкалы во всех бенчмарках | Добавить EV/EBITDA для Financial (через P/B или CET1 ratio) |
