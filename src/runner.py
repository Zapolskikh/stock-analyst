"""
Batch runner — runs the full analysis pipeline over a list of tickers,
applies an optional AI filter, and saves per-stock reports + price charts.

Workflow
--------
1. Analyse every ticker in *tickers*.
2. Screen by a minimum overall score threshold (default 65).
3. Pass screened stocks to the AI connector for a consistency check.
4. Save full report (.txt) and interactive price chart (.html) for every
   stock that passes the AI filter.
5. Return a ranked list of BatchResult objects for further use.

Usage::

    from src.runner import run_universe, BatchResult
    from src.ai.connector import build_connector

    connector = build_connector("ollama", model="llama3.1")
    # connector = build_connector("claude", api_key="sk-...")
    # connector = build_connector("null")          # skip AI, keep all

    results = run_universe(
        tickers=["NVDA", "AAPL", "MSFT", "TSLA"],
        connector=connector,
        output_dir="output/batch",
        min_score=65.0,
    )

    for r in results:
        print(r.summary_line())
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.ai.connector import AIConnector, AIInput, AIReview, build_connector
from src.engine.engine import AnalysisResult, analyse
from src.output.formatter import format_report, format_brief


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    """Per-ticker outcome from a batch run."""
    ticker: str
    result: Optional[AnalysisResult]       # None when analysis failed
    error: Optional[str]                   # set when analysis raised

    ai_review: Optional[AIReview] = None   # set when AI was invoked
    ai_skipped: bool = False               # True when score < min_score
    report_path: Optional[Path] = None     # saved report file
    chart_path: Optional[Path] = None      # saved price chart HTML

    # Convenience ----------------------------------------------------------

    @property
    def passed_screen(self) -> bool:
        """True when stock cleared the score threshold and AI (if used) agreed."""
        if self.result is None:
            return False
        if self.ai_skipped:
            return False
        if self.ai_review is not None:
            # NullConnector (no AI) — abstain means "no opinion", don't block
            is_null = getattr(self.ai_review, "backend", "") == "null"
            if not is_null:
                # Real AI: block on explicit rejection or abstain (uncertain)
                if self.ai_review.abstain or not self.ai_review.agreement:
                    return False
        return True

    def summary_line(self) -> str:
        """One-line console summary."""
        if self.result is None:
            return f"  {self.ticker:<6}  ERROR: {self.error}"

        r = self.result
        ai_tag = ""
        if self.ai_review is not None:
            if self.ai_review.abstain:
                ai_tag = " [AI:abstain]"
            elif self.ai_review.agreement:
                ai_tag = " [AI:✓]"
            else:
                ai_tag = " [AI:✗]"
        elif self.ai_skipped:
            ai_tag = " [skipped]"

        chart_tag = f"  chart→{self.chart_path.name}" if self.chart_path else ""
        return (
            f"  {r.ticker:<6}  {r.company_type.value:<22}  "
            f"score={r.overall_score:4.1f}  {r.decision:<5}{ai_tag}{chart_tag}"
        )


# ---------------------------------------------------------------------------
# Core batch runner
# ---------------------------------------------------------------------------

def run_universe(
    tickers: list[str],
    connector: Optional[AIConnector] = None,
    output_dir: str | Path = "output/batch",
    min_score: float = 65.0,
    save_all_reports: bool = False,
    save_charts: bool = True,
) -> list[BatchResult]:
    """Run the full analysis pipeline over *tickers*.

    Parameters
    ----------
    tickers:
        List of stock ticker symbols to analyse.
    connector:
        AI backend to use. Pass ``build_connector("null")`` to skip AI
        (all stocks that meet *min_score* are kept). Defaults to NullConnector.
    output_dir:
        Directory where reports and charts are saved.
    min_score:
        Overall score threshold (0–100). Only stocks at or above this score
        are sent to the AI filter and saved.
    save_all_reports:
        When True, save text reports for every stock regardless of score.
        When False (default), save only for those that pass the screen.
    save_charts:
        Save an interactive price chart HTML for every stock that passes
        the AI filter (or every stock above min_score when using NullConnector).

    Returns
    -------
    list[BatchResult]
        Sorted descending by overall score (failed analyses last).
    """
    if connector is None:
        connector = build_connector("null")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    batch: list[BatchResult] = []

    total = len(tickers)
    for idx, ticker in enumerate(tickers, 1):
        ticker = ticker.upper().strip()
        print(f"[{idx}/{total}] Analysing {ticker}...", end=" ", flush=True)

        # ── Step 1: run analysis ───────────────────────────────────────────
        try:
            result = analyse(ticker)
        except Exception as exc:
            print(f"ERROR — {exc}")
            batch.append(BatchResult(ticker=ticker, result=None, error=str(exc)))
            continue

        br = BatchResult(ticker=ticker, result=result, error=None)

        score_tag = f"score={result.overall_score:.1f} ({result.decision})"

        # ── Step 2: score threshold filter ────────────────────────────────
        if result.overall_score < min_score:
            print(f"{score_tag}  → below threshold ({min_score:.0f}), skipped")
            br.ai_skipped = True
            if save_all_reports:
                br.report_path = _save_report(result, out)
            batch.append(br)
            continue

        # ── Step 3: AI consistency check ──────────────────────────────────
        try:
            ai_input = AIInput.from_result(result)
            ai_review = connector.review(ai_input)
            br.ai_review = ai_review
            ai_status = "AI:abstain" if ai_review.abstain else ("AI:✓" if ai_review.agreement else "AI:✗")
        except Exception as exc:
            ai_review = None
            ai_status = f"AI:ERROR({exc})"

        print(f"{score_tag}  → {ai_status}", end="")

        # ── Step 4: save report ───────────────────────────────────────────
        if save_all_reports or br.passed_screen:
            br.report_path = _save_report(result, out)

        # ── Step 5: save price chart ──────────────────────────────────────
        if save_charts and br.passed_screen:
            try:
                br.chart_path = _save_price_chart(result, out)
                print(f"  chart→{br.chart_path.name}", end="")
            except Exception as exc:
                print(f"  chart:ERROR({exc})", end="")
            try:
                _save_fundamental_charts(result, out)
            except Exception as exc:
                print(f"  fund_charts:ERROR({exc})", end="")

        print()
        batch.append(br)

    # Sort: passed first (by score desc), then skipped, then errors
    batch.sort(
        key=lambda b: (
            0 if b.passed_screen else (1 if b.result is not None else 2),
            -(b.result.overall_score if b.result else 0),
        )
    )
    return batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_report(result: AnalysisResult, out_dir: Path) -> Path:
    """Write the full text report to *out_dir/<TICKER>_report.txt*."""
    path = out_dir / f"{result.ticker}_report.txt"
    path.write_text(format_report(result), encoding="utf-8")
    return path


def _save_price_chart(result: AnalysisResult, out_dir: Path) -> Path:
    """Fetch OHLCV and save an interactive price chart to *out_dir/<TICKER>_chart.html*."""
    from src.data.price import fetch_ohlcv
    from src.charts.price_chart import build_price_chart

    df = fetch_ohlcv(result.ticker, period="2y")
    fig = build_price_chart(df, result.ticker)
    path = out_dir / f"{result.ticker}_chart.html"
    fig.write_html(str(path))
    return path


def _save_fundamental_charts(result: AnalysisResult, out_dir: Path) -> list[Path]:
    """Build and save fundamental HTML charts (revenue, margins, balance sheet, etc.)."""
    from src.data.sec_edgar import fetch_fundamentals
    from src.charts.fundamental_chart import build_fundamental_charts

    fundamentals = fetch_fundamentals(result.ticker)
    saved = build_fundamental_charts(fundamentals, result.ticker, out_dir)
    return [p for _, p in saved]


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_batch_summary(batch: list[BatchResult]) -> None:
    """Print a formatted summary table of all batch results."""
    passed   = [b for b in batch if b.passed_screen]
    screened = [b for b in batch if not b.passed_screen and b.result is not None and not b.ai_skipped]
    skipped  = [b for b in batch if b.ai_skipped]
    errors   = [b for b in batch if b.result is None]

    sep = "─" * 72

    print(f"\n{sep}")
    print(f"  BATCH RESULTS  ({len(batch)} tickers analysed)")
    print(sep)

    if passed:
        print(f"\n  ✅ PASSED  ({len(passed)} stocks)")
        for b in passed:
            print(b.summary_line())
            if b.ai_review and b.ai_review.narrative:
                # Truncate narrative to one line for console display
                narrative = b.ai_review.narrative.splitlines()[0][:90]
                print(f"         AI: {narrative}")
            if b.report_path:
                print(f"         report → {b.report_path}")

    if screened:
        print(f"\n  ⚠️  AI REJECTED  ({len(screened)} stocks)")
        for b in screened:
            print(b.summary_line())

    if skipped:
        print(f"\n  ⏭  BELOW THRESHOLD  ({len(skipped)} stocks)")
        for b in skipped:
            print(b.summary_line())

    if errors:
        print(f"\n  ❌ ERRORS  ({len(errors)} stocks)")
        for b in errors:
            print(b.summary_line())

    print(f"\n{sep}\n")
