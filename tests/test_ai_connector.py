"""Tests for src/ai/connector.py"""
from __future__ import annotations

import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from src.ai.connector import (
    AIConnector,
    AIInput,
    AIReview,
    ClaudeConnector,
    NullConnector,
    OllamaConnector,
    build_connector,
)
from src.data.normalizer import NormalisedData
from src.engine.engine import analyse_nd

# ---------------------------------------------------------------------------
# Helper: minimal NormalisedData
# ---------------------------------------------------------------------------

def _nd(**kwargs) -> NormalisedData:
    base = dict(
        ticker="TEST",
        years=[2020, 2021, 2022, 2023],
        revenue_annual=[200e9, 250e9, 300e9, 350e9],
        gross_profit_annual=[100e9, 130e9, 160e9, 200e9],
        operating_income_annual=[50e9, 70e9, 90e9, 110e9],
        net_income_annual=[40e9, 55e9, 70e9, 85e9],
        operating_cf_annual=[60e9, 80e9, 100e9, 120e9],
        capex_annual=[10e9, 12e9, 14e9, 16e9],
        equity_annual=[80e9, 90e9, 100e9, 110e9],
        total_assets_annual=[200e9, 220e9, 240e9, 260e9],
        total_liabilities_annual=[120e9, 130e9, 140e9, 150e9],
        long_term_debt_annual=[50e9, 45e9, 40e9, 35e9],
        eps_diluted_annual=[5.0, 7.0, 9.0, 11.0],
        rd_expense_annual=[5e9, 6e9, 7e9, 8e9],
        fcf_annual=[50e9, 68e9, 86e9, 104e9],
        gross_margin_annual=[50.0, 52.0, 53.3, 57.1],
        operating_margin_annual=[25.0, 28.0, 30.0, 31.4],
        net_margin_annual=[20.0, 22.0, 23.3, 24.3],
        revenue_growth_annual=[float("nan"), 25.0, 20.0, 16.7],
        eps_growth_annual=[float("nan"), 40.0, 28.6, 22.2],
        roe_annual=[50.0, 61.1, 70.0, 77.3],
        roa_annual=[20.0, 25.0, 29.2, 32.7],
        debt_to_equity_annual=[0.625, 0.5, 0.4, 0.318],
        current_price=155.0,
        market_cap=2.4e12,
        pe_trailing=28.5,
        pe_forward=24.0,
        beta=1.2,
        sector="Technology",
        industry="Semiconductors",
        dividend_yield=0.002,
        years_of_history=4,
        data_quality="good",
        close_prices=[float(100 + i * 0.5) for i in range(252)],
        spy_close_prices=[],
        shares_dilution_annual=[float("nan"), 1.0, 1.2, 0.8],
        ebitda_annual=[],
        cash_annual=[],
    )
    base.update(kwargs)
    return NormalisedData(**base)


_VALID_JSON_RESPONSE = json.dumps({
    "agreement": True,
    "contradictions": [],
    "confidence": 0.85,
    "narrative": "The analysis appears internally consistent.",
    "action": "watch",
    "abstain": False,
})


# ---------------------------------------------------------------------------
# AIInput
# ---------------------------------------------------------------------------

class TestAIInput(unittest.TestCase):

    def setUp(self):
        self.result = analyse_nd(_nd())
        self.ai_input = AIInput.from_result(self.result)

    def test_symbol_matches_ticker(self):
        self.assertEqual(self.ai_input.symbol, "TEST")

    def test_block_scores_are_floats(self):
        for k, v in self.ai_input.block_scores.items():
            self.assertIsInstance(v, float, f"{k} should be float")

    def test_horizon_scores_present(self):
        self.assertIn("short", self.ai_input.horizon_scores)
        self.assertIn("medium", self.ai_input.horizon_scores)
        self.assertIn("long", self.ai_input.horizon_scores)

    def test_horizon_values_in_range(self):
        for k, v in self.ai_input.horizon_scores.items():
            self.assertGreaterEqual(v, 0.0, k)
            self.assertLessEqual(v, 100.0, k)

    def test_company_type_is_string(self):
        self.assertIsInstance(self.ai_input.company_type, str)

    def test_data_quality_is_string(self):
        self.assertIn(self.ai_input.data_quality, {"good", "partial", "poor"})

    def test_to_json_is_valid_json(self):
        text = self.ai_input.to_json()
        data = json.loads(text)
        self.assertIn("symbol", data)

    def test_to_dict_roundtrip(self):
        d = self.ai_input.to_dict()
        self.assertEqual(d["symbol"], self.ai_input.symbol)
        self.assertEqual(d["block_scores"], self.ai_input.block_scores)

    def test_critical_stop_factors_empty_for_healthy_stock(self):
        # _nd() has no critical stop factors
        self.assertIsInstance(self.ai_input.critical_stop_factors, list)

    def test_top_features_are_lists(self):
        self.assertIsInstance(self.ai_input.top_positive_features, list)
        self.assertIsInstance(self.ai_input.top_negative_features, list)

    def test_critical_stop_factors_captured(self):
        # Stock with extreme debt triggers a critical stop
        nd = _nd(debt_to_equity_annual=[5.0, 6.0, 7.0, 8.0])
        result = analyse_nd(nd)
        ai_in = AIInput.from_result(result)
        # At D/E=8 we expect "High Debt" critical stop factor
        self.assertIn("High Debt", ai_in.critical_stop_factors)

    def test_features_contain_block_prefix(self):
        # Features should be in "block.metric=score" format
        for f in self.ai_input.top_positive_features + self.ai_input.top_negative_features:
            self.assertIn("=", f, f"feature '{f}' should contain '='")
            self.assertIn(".", f, f"feature '{f}' should contain 'block.metric' dot")


# ---------------------------------------------------------------------------
# AIReview post-init validation
# ---------------------------------------------------------------------------

class TestAIReview(unittest.TestCase):

    def test_confidence_clamped_high(self):
        r = AIReview(True, [], 5.0, "ok", "buy", False)
        self.assertEqual(r.confidence, 1.0)

    def test_confidence_clamped_low(self):
        r = AIReview(False, [], -1.0, "bad", "avoid", True)
        self.assertEqual(r.confidence, 0.0)

    def test_action_lowercased(self):
        r = AIReview(True, [], 0.7, "ok", "BUY", False)
        self.assertEqual(r.action, "buy")

    def test_invalid_action_defaults_to_hold(self):
        r = AIReview(True, [], 0.5, "ok", "strong_buy", False)
        self.assertEqual(r.action, "hold")

    def test_valid_actions_accepted(self):
        for action in ("buy", "watch", "hold", "avoid"):
            r = AIReview(True, [], 0.5, "ok", action, False)
            self.assertEqual(r.action, action)

    def test_backend_defaults_to_unknown(self):
        r = AIReview(True, [], 0.5, "ok", "hold", False)
        self.assertEqual(r.backend, "unknown")


# ---------------------------------------------------------------------------
# NullConnector
# ---------------------------------------------------------------------------

class TestNullConnector(unittest.TestCase):

    def setUp(self):
        self.connector = NullConnector()
        self.ai_input = AIInput.from_result(analyse_nd(_nd()))

    def test_returns_airview(self):
        review = self.connector.review(self.ai_input)
        self.assertIsInstance(review, AIReview)

    def test_always_abstains(self):
        review = self.connector.review(self.ai_input)
        self.assertTrue(review.abstain)

    def test_backend_is_null(self):
        review = self.connector.review(self.ai_input)
        self.assertEqual(review.backend, "null")

    def test_no_contradictions(self):
        review = self.connector.review(self.ai_input)
        self.assertEqual(review.contradictions, [])

    def test_confidence_is_zero(self):
        review = self.connector.review(self.ai_input)
        self.assertEqual(review.confidence, 0.0)


# ---------------------------------------------------------------------------
# _parse_response (via a concrete subclass)
# ---------------------------------------------------------------------------

class _ConcreteConnector(AIConnector):
    """Minimal concrete connector for testing _parse_response."""
    def review(self, ai_input: AIInput) -> AIReview:
        return self._parse_response("{}", "test")


class TestParseResponse(unittest.TestCase):

    def setUp(self):
        self.conn = _ConcreteConnector()

    def test_valid_json_parsed(self):
        r = self.conn._parse_response(_VALID_JSON_RESPONSE, "test")
        self.assertTrue(r.agreement)
        self.assertAlmostEqual(r.confidence, 0.85)
        self.assertEqual(r.action, "watch")
        self.assertFalse(r.abstain)
        self.assertEqual(r.backend, "test")

    def test_markdown_fenced_json_parsed(self):
        text = f"```json\n{_VALID_JSON_RESPONSE}\n```"
        r = self.conn._parse_response(text, "test")
        self.assertTrue(r.agreement)

    def test_invalid_json_returns_abstain(self):
        r = self.conn._parse_response("not json at all", "test")
        self.assertTrue(r.abstain)
        self.assertEqual(r.confidence, 0.0)
        self.assertTrue(len(r.contradictions) > 0)

    def test_missing_keys_use_defaults(self):
        r = self.conn._parse_response("{}", "test")
        self.assertFalse(r.agreement)
        self.assertEqual(r.action, "hold")
        self.assertFalse(r.abstain)

    def test_backend_label_preserved(self):
        r = self.conn._parse_response(_VALID_JSON_RESPONSE, "claude/claude-opus-4-5")
        self.assertEqual(r.backend, "claude/claude-opus-4-5")


# ---------------------------------------------------------------------------
# ClaudeConnector
# ---------------------------------------------------------------------------

class TestClaudeConnector(unittest.TestCase):

    def _make_mock_anthropic(self):
        """Build a mock anthropic module."""
        mock_anthropic = types.ModuleType("anthropic")
        mock_client = MagicMock()
        mock_anthropic.Anthropic = MagicMock(return_value=mock_client)
        return mock_anthropic, mock_client

    def test_import_error_without_package(self):
        with patch.dict(sys.modules, {"anthropic": None}):
            with self.assertRaises(ImportError) as ctx:
                ClaudeConnector(api_key="key")
            self.assertIn("pip install anthropic", str(ctx.exception))

    def test_review_calls_api_and_returns_airview(self):
        mock_anthropic, mock_client = self._make_mock_anthropic()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=_VALID_JSON_RESPONSE)]
        mock_message.usage.input_tokens = 1000
        mock_message.usage.output_tokens = 200
        mock_client.messages.create.return_value = mock_message

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            conn = ClaudeConnector(api_key="fake-key", model="claude-opus-4-5")
            ai_input = AIInput.from_result(analyse_nd(_nd()))
            review = conn.review(ai_input)

        self.assertIsInstance(review, AIReview)
        self.assertEqual(review.backend, "claude/claude-opus-4-5")
        mock_client.messages.create.assert_called_once()

    def test_review_passes_system_prompt(self):
        mock_anthropic, mock_client = self._make_mock_anthropic()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=_VALID_JSON_RESPONSE)]
        mock_message.usage.input_tokens = 1000
        mock_message.usage.output_tokens = 200
        mock_client.messages.create.return_value = mock_message

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            conn = ClaudeConnector(api_key="fake-key")
            ai_input = AIInput.from_result(analyse_nd(_nd()))
            conn.review(ai_input)

        call_kwargs = mock_client.messages.create.call_args
        self.assertIn("system", call_kwargs.kwargs)

    def test_api_error_propagates(self):
        mock_anthropic, mock_client = self._make_mock_anthropic()
        mock_client.messages.create.side_effect = RuntimeError("API error")

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            conn = ClaudeConnector(api_key="fake-key")
            ai_input = AIInput.from_result(analyse_nd(_nd()))
            with self.assertRaises(RuntimeError):
                conn.review(ai_input)

    def test_model_included_in_backend_label(self):
        mock_anthropic, mock_client = self._make_mock_anthropic()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=_VALID_JSON_RESPONSE)]
        mock_message.usage.input_tokens = 1000
        mock_message.usage.output_tokens = 200
        mock_client.messages.create.return_value = mock_message

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            conn = ClaudeConnector(api_key="k", model="claude-haiku-3-5")
            ai_input = AIInput.from_result(analyse_nd(_nd()))
            review = conn.review(ai_input)

        self.assertIn("claude-haiku-3-5", review.backend)


# ---------------------------------------------------------------------------
# OllamaConnector
# ---------------------------------------------------------------------------

class TestOllamaConnector(unittest.TestCase):

    def _mock_urlopen(self, response_body: dict):
        """Return a context manager mock that yields a fake HTTP response."""
        raw = json.dumps(response_body).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = raw
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_review_returns_airview(self):
        ollama_response = {"message": {"content": _VALID_JSON_RESPONSE}}
        mock_resp = self._mock_urlopen(ollama_response)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            conn = OllamaConnector(model="llama3.1")
            ai_input = AIInput.from_result(analyse_nd(_nd()))
            review = conn.review(ai_input)

        self.assertIsInstance(review, AIReview)
        self.assertIn("ollama", review.backend)
        self.assertIn("llama3.1", review.backend)

    def test_backend_label_includes_model(self):
        ollama_response = {"message": {"content": _VALID_JSON_RESPONSE}}
        mock_resp = self._mock_urlopen(ollama_response)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            conn = OllamaConnector(model="mistral")
            ai_input = AIInput.from_result(analyse_nd(_nd()))
            review = conn.review(ai_input)

        self.assertEqual(review.backend, "ollama/mistral")

    def test_connection_error_when_unavailable(self):
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            conn = OllamaConnector()
            ai_input = AIInput.from_result(analyse_nd(_nd()))
            with self.assertRaises(ConnectionError):
                conn.review(ai_input)

    def test_posts_to_api_chat_endpoint(self):
        ollama_response = {"message": {"content": _VALID_JSON_RESPONSE}}
        mock_resp = self._mock_urlopen(ollama_response)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            conn = OllamaConnector(base_url="http://localhost:11434")
            ai_input = AIInput.from_result(analyse_nd(_nd()))
            conn.review(ai_input)

        call_args = mock_open.call_args
        req = call_args[0][0]
        self.assertIn("/api/chat", req.full_url)

    def test_request_contains_model_name(self):
        ollama_response = {"message": {"content": _VALID_JSON_RESPONSE}}
        mock_resp = self._mock_urlopen(ollama_response)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            conn = OllamaConnector(model="gemma2")
            ai_input = AIInput.from_result(analyse_nd(_nd()))
            conn.review(ai_input)

        call_args = mock_open.call_args
        req = call_args[0][0]
        payload = json.loads(req.data.decode())
        self.assertEqual(payload["model"], "gemma2")


# ---------------------------------------------------------------------------
# build_connector factory
# ---------------------------------------------------------------------------

class TestBuildConnector(unittest.TestCase):

    def test_build_null(self):
        conn = build_connector("null")
        self.assertIsInstance(conn, NullConnector)

    def test_build_unknown_raises(self):
        with self.assertRaises(ValueError) as ctx:
            build_connector("openai")
        self.assertIn("openai", str(ctx.exception))

    def test_build_null_ignores_kwargs(self):
        # NullConnector accepts no kwargs — factory should pass kwargs only to others
        conn = build_connector("null")
        self.assertIsInstance(conn, NullConnector)

    def test_build_ollama(self):
        conn = build_connector("ollama", model="mistral", base_url="http://localhost:11434")
        self.assertIsInstance(conn, OllamaConnector)
        self.assertEqual(conn._model, "mistral")

    def test_build_ollama_defaults(self):
        conn = build_connector("ollama")
        self.assertIsInstance(conn, OllamaConnector)
        self.assertEqual(conn._model, "llama3.1")

    def test_build_claude_raises_without_anthropic(self):
        with patch.dict(sys.modules, {"anthropic": None}):
            with self.assertRaises(ImportError):
                build_connector("claude", api_key="fake")

    def test_build_claude_with_mock_package(self):
        mock_anthropic = types.ModuleType("anthropic")
        mock_anthropic.Anthropic = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            conn = build_connector("claude", api_key="fake-key")

        self.assertIsInstance(conn, ClaudeConnector)


if __name__ == "__main__":
    unittest.main()
