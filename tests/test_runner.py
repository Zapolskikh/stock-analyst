"""
Tests for src/runner.py — batch analysis pipeline.

Uses only NullConnector and mocked analyse() calls (no network).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.ai.connector import build_connector
from src.runner import BatchResult, _save_report, print_batch_summary, run_universe

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_result(ticker: str, score: float, decision: str = "Buy") -> MagicMock:
    """Create a minimal AnalysisResult mock."""
    result = MagicMock()
    result.ticker = ticker
    result.overall_score = score
    result.decision = decision
    result.company_type.value = "Mature Tech"
    result.data_confidence = "good"
    result.block_scores = {}
    result.stop_factors = []
    result.horizon.short = score * 0.9
    result.horizon.medium = score
    result.horizon.long = score * 1.05
    result.horizon_decisions.short = decision
    result.horizon_decisions.medium = decision
    result.horizon_decisions.long = decision
    result.classification_confidence = 0.85
    result.rating = "Good Candidate"
    result.config_version = "1.0.0"
    result.nd = None  # prevent _extract_raw_metrics from running on a MagicMock
    result.trade_rec = None
    result.current_price = 100.0
    return result


# ---------------------------------------------------------------------------
# BatchResult unit tests
# ---------------------------------------------------------------------------

class TestBatchResult:
    def test_passed_screen_when_no_ai(self):
        r = _make_result("AAPL", 75.0)
        br = BatchResult(ticker="AAPL", result=r, error=None)
        assert br.passed_screen is True

    def test_not_passed_when_ai_skipped(self):
        r = _make_result("AAPL", 75.0)
        br = BatchResult(ticker="AAPL", result=r, error=None, ai_skipped=True)
        assert br.passed_screen is False

    def test_not_passed_when_result_none(self):
        br = BatchResult(ticker="AAPL", result=None, error="network error")
        assert br.passed_screen is False

    def test_not_passed_when_ai_disagrees(self):
        from src.ai.connector import AIReview
        r = _make_result("TSLA", 72.0)
        review = AIReview(agreement=False, contradictions=["high D/E vs good risk score"],
                          confidence=0.7, narrative="Inconsistent.", action="hold", abstain=False)
        br = BatchResult(ticker="TSLA", result=r, error=None, ai_review=review)
        assert br.passed_screen is False

    def test_not_passed_when_ai_abstains(self):
        from src.ai.connector import AIReview
        r = _make_result("XYZ", 68.0)
        review = AIReview(agreement=True, contradictions=[], confidence=0.2,
                          narrative="Poor data.", action="hold", abstain=True)
        br = BatchResult(ticker="XYZ", result=r, error=None, ai_review=review)
        assert br.passed_screen is False

    def test_passed_when_ai_agrees(self):
        from src.ai.connector import AIReview
        r = _make_result("MSFT", 80.0)
        review = AIReview(agreement=True, contradictions=[], confidence=0.88,
                          narrative="Consistent.", action="buy", abstain=False)
        br = BatchResult(ticker="MSFT", result=r, error=None, ai_review=review)
        assert br.passed_screen is True

    def test_summary_line_error(self):
        br = BatchResult(ticker="BAD", result=None, error="timeout")
        line = br.summary_line()
        assert "BAD" in line
        assert "ERROR" in line
        assert "timeout" in line

    def test_summary_line_passed(self):
        r = _make_result("NVDA", 85.0)
        br = BatchResult(ticker="NVDA", result=r, error=None)
        line = br.summary_line()
        assert "NVDA" in line
        assert "85.0" in line

    def test_summary_line_ai_confirmed(self):
        from src.ai.connector import AIReview
        r = _make_result("AAPL", 77.0)
        review = AIReview(agreement=True, contradictions=[], confidence=0.9,
                          narrative="Good.", action="buy", abstain=False)
        br = BatchResult(ticker="AAPL", result=r, error=None, ai_review=review)
        line = br.summary_line()
        assert "AI:✓" in line


# ---------------------------------------------------------------------------
# run_universe integration tests (mocked analyse)
# ---------------------------------------------------------------------------

class TestRunUniverse:
    def _patch_analyse(self, side_effects: dict[str, float | Exception]):
        """Return a mock that returns results keyed by ticker."""
        def fake_analyse(ticker):
            v = side_effects[ticker]
            if isinstance(v, Exception):
                raise v
            return _make_result(ticker, v)
        return fake_analyse

    def test_all_below_threshold_are_skipped(self, tmp_path):
        scores = {"AAPL": 50.0, "MSFT": 55.0}
        with patch("src.runner.analyse", side_effect=self._patch_analyse(scores)):
            connector = build_connector("null")
            results = run_universe(["AAPL", "MSFT"], connector=connector,
                                   output_dir=tmp_path, min_score=65.0,
                                   save_charts=False)
        assert all(b.ai_skipped for b in results)
        assert not any(b.passed_screen for b in results)

    def test_above_threshold_passes_with_null_connector(self, tmp_path):
        scores = {"AAPL": 75.0, "MSFT": 80.0}
        with patch("src.runner.analyse", side_effect=self._patch_analyse(scores)):
            connector = build_connector("null")
            results = run_universe(["AAPL", "MSFT"], connector=connector,
                                   output_dir=tmp_path, min_score=65.0,
                                   save_charts=False)
        # NullConnector abstains — so passed_screen is False (AI abstained)
        for b in results:
            assert b.ai_review is not None
            assert b.ai_review.abstain is True

    def test_error_ticker_captured(self, tmp_path):
        def fake_analyse(ticker):
            raise RuntimeError("No data")
        with patch("src.runner.analyse", side_effect=fake_analyse):
            connector = build_connector("null")
            results = run_universe(["BAD"], connector=connector,
                                   output_dir=tmp_path, save_charts=False)
        assert len(results) == 1
        assert results[0].result is None
        assert results[0].error == "No data"

    def test_results_sorted_passed_first(self, tmp_path):
        """Passed stocks appear before skipped/error stocks."""
        from src.ai.connector import AIReview, NullConnector

        class AlwaysAgreeConnector(NullConnector):
            def review(self, ai_input):
                return AIReview(agreement=True, contradictions=[], confidence=0.9,
                                narrative="OK", action="buy", abstain=False)

        scores = {"AAPL": 80.0, "MSFT": 70.0, "LOW": 40.0}
        with patch("src.runner.analyse", side_effect=self._patch_analyse(scores)):
            results = run_universe(
                ["AAPL", "MSFT", "LOW"],
                connector=AlwaysAgreeConnector(),
                output_dir=tmp_path,
                min_score=65.0,
                save_charts=False,
            )
        # LOW is below threshold → ai_skipped; AAPL and MSFT pass
        passed = [b for b in results if b.passed_screen]
        skipped = [b for b in results if b.ai_skipped]
        assert len(passed) == 2
        assert len(skipped) == 1
        # Passed stocks appear before skipped
        passed_indices = [i for i, b in enumerate(results) if b.passed_screen]
        skipped_indices = [i for i, b in enumerate(results) if b.ai_skipped]
        assert max(passed_indices) < min(skipped_indices)

    def test_report_saved_for_passing_stock(self, tmp_path):
        from src.ai.connector import AIReview, NullConnector

        class AlwaysAgreeConnector(NullConnector):
            def review(self, ai_input):
                return AIReview(agreement=True, contradictions=[], confidence=0.9,
                                narrative="OK", action="buy", abstain=False)

        scores = {"NVDA": 85.0}
        with patch("src.runner.analyse", side_effect=self._patch_analyse(scores)):
            with patch("src.runner._save_price_chart", return_value=tmp_path / "NVDA_chart.html"):
                results = run_universe(
                    ["NVDA"],
                    connector=AlwaysAgreeConnector(),
                    output_dir=tmp_path,
                    min_score=65.0,
                    save_charts=True,
                )
        b = results[0]
        assert b.passed_screen is True
        assert b.report_path is not None
        assert b.report_path.exists()

    def test_save_all_reports_saves_below_threshold(self, tmp_path):
        scores = {"AAPL": 50.0}
        with patch("src.runner.analyse", side_effect=self._patch_analyse(scores)):
            results = run_universe(
                ["AAPL"],
                connector=build_connector("null"),
                output_dir=tmp_path,
                min_score=65.0,
                save_all_reports=True,
                save_charts=False,
            )
        b = results[0]
        assert b.report_path is not None
        assert b.report_path.exists()

    def test_output_dir_created(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "dir"
        scores = {"AAPL": 50.0}
        with patch("src.runner.analyse", side_effect=self._patch_analyse(scores)):
            run_universe(["AAPL"], output_dir=out, save_charts=False)
        assert out.exists()


# ---------------------------------------------------------------------------
# _save_report
# ---------------------------------------------------------------------------

class TestSaveReport:
    def test_creates_file(self, tmp_path):
        result = _make_result("NVDA", 85.0)
        path = _save_report(result, tmp_path)
        assert path.exists()
        assert path.name == "NVDA_report.txt"

    def test_file_not_empty(self, tmp_path):
        result = _make_result("AAPL", 75.0)
        path = _save_report(result, tmp_path)
        content = path.read_text(encoding="utf-8")
        assert len(content) > 0


# ---------------------------------------------------------------------------
# print_batch_summary smoke test
# ---------------------------------------------------------------------------

class TestPrintBatchSummary:
    def test_runs_without_error(self, capsys):
        batch = [
            BatchResult(ticker="AAPL", result=_make_result("AAPL", 80.0), error=None),
            BatchResult(ticker="BAD",  result=None, error="timeout"),
        ]
        print_batch_summary(batch)
        out = capsys.readouterr().out
        assert "AAPL" in out
        assert "BAD" in out
