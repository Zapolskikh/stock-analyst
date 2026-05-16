"""AI connector module — consistency-checker layer over AnalysisResult.

Supported backends:
  - "null"   — always abstains; used when no AI is configured (default)
  - "claude" — Anthropic Claude API (requires ``anthropic`` package)
  - "ollama" — local Ollama HTTP API (no extra packages required)

Usage::

    from src.ai.connector import build_connector, AIInput, AIReview

    # Build snapshot from a completed analysis
    ai_input  = AIInput.from_result(result)

    # Pick a backend
    connector = build_connector("null")
    connector = build_connector("claude", api_key=os.environ["ANTHROPIC_API_KEY"])
    connector = build_connector("ollama", model="llama3.1")

    # Run consistency check
    review = connector.review(ai_input)
    print(review.agreement, review.narrative)

The AI verifier does NOT replace the quantitative scoring engine.
Its role is consistency-checking and human-readable narrative.
"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.data.normalizer import NormalisedData
    from src.engine.engine import AnalysisResult


# ---------------------------------------------------------------------------
# Raw metrics extractor
# ---------------------------------------------------------------------------

def _r(val: Optional[float], decimals: int = 2) -> Optional[float]:
    """Round optional float, return None if missing."""
    return round(val, decimals) if val is not None else None


def _extract_raw_metrics(nd: "NormalisedData") -> dict:
    """Extract key raw financial metrics from NormalisedData for AI cross-check.

    Organised into sections that map to the scoring blocks so the AI can
    independently verify whether our 0-10 scores are plausible.
    """
    metrics: dict = {}

    # ── Valuation ──────────────────────────────────────────────────────────
    val: dict = {}
    if nd.pe_trailing is not None:
        val["pe_trailing"] = _r(nd.pe_trailing, 1)
    if nd.pe_forward is not None:
        val["pe_forward"] = _r(nd.pe_forward, 1)
    if nd.pe_hist_avg is not None:
        val["pe_hist_avg"] = _r(nd.pe_hist_avg, 1)
    if nd.market_cap is not None and nd.ttm_revenue and nd.ttm_revenue > 0:
        val["ps_ttm"] = _r(nd.market_cap / nd.ttm_revenue, 1)
    if nd.analyst_target_median and nd.current_price and nd.current_price > 0:
        upside = (nd.analyst_target_median / nd.current_price - 1) * 100
        val["analyst_target_median"] = _r(nd.analyst_target_median, 2)
        val["analyst_upside_pct"] = _r(upside, 1)
    if nd.analyst_count is not None:
        val["analyst_count"] = nd.analyst_count
    if nd.recommendation_key:
        val["analyst_consensus"] = nd.recommendation_key
    if val:
        metrics["valuation"] = val

    # ── Quality / Growth ───────────────────────────────────────────────────
    qual: dict = {}
    # Most recent YoY revenue growth
    if nd.revenue_growth_annual:
        valid_growth = [g for g in nd.revenue_growth_annual if g is not None]
        if valid_growth:
            qual["revenue_growth_1y_pct"] = _r(valid_growth[-1], 1)
            if len(valid_growth) >= 3:
                qual["revenue_growth_3y_avg_pct"] = _r(
                    sum(valid_growth[-3:]) / 3, 1
                )
    # TTM margins (preferred) or latest annual
    if nd.ttm_gross_profit and nd.ttm_revenue and nd.ttm_revenue > 0:
        qual["gross_margin_pct"] = _r(nd.ttm_gross_profit / nd.ttm_revenue * 100, 1)
    elif nd.gross_margin_annual:
        valid = [x for x in nd.gross_margin_annual if x is not None]
        if valid:
            qual["gross_margin_pct"] = _r(valid[-1], 1)
    if nd.ttm_net_income and nd.ttm_revenue and nd.ttm_revenue > 0:
        qual["net_margin_pct"] = _r(nd.ttm_net_income / nd.ttm_revenue * 100, 1)
    elif nd.net_margin_annual:
        valid = [x for x in nd.net_margin_annual if x is not None]
        if valid:
            qual["net_margin_pct"] = _r(valid[-1], 1)
    if nd.roe_annual:
        valid = [x for x in nd.roe_annual if x is not None]
        if valid:
            roe = valid[-1]
            # Cap extreme values (buyback companies can show 1000%+ ROE)
            if abs(roe) < 500:
                qual["roe_pct"] = _r(roe, 1)
            else:
                qual["roe_pct_note"] = "extreme (equity near zero due to buybacks)"
    # FCF yield (TTM FCF / market cap)
    if nd.ttm_fcf and nd.market_cap and nd.market_cap > 0:
        qual["fcf_yield_pct"] = _r(nd.ttm_fcf / nd.market_cap * 100, 1)
    if qual:
        metrics["quality"] = qual

    # ── Technical ──────────────────────────────────────────────────────────
    tech: dict = {}
    if nd.close_prices and len(nd.close_prices) >= 50:
        price = nd.close_prices[-1]
        ma50  = sum(nd.close_prices[-50:]) / 50
        tech["price_vs_ma50_pct"] = _r((price / ma50 - 1) * 100, 1)
        if len(nd.close_prices) >= 200:
            ma200 = sum(nd.close_prices[-200:]) / 200
            tech["price_vs_ma200_pct"] = _r((price / ma200 - 1) * 100, 1)
    if nd.atr_pct is not None:
        tech["atr_pct_daily"] = _r(nd.atr_pct, 2)
    if tech:
        metrics["technical"] = tech

    # ── Risk ───────────────────────────────────────────────────────────────
    risk: dict = {}
    if nd.beta is not None:
        risk["beta"] = _r(nd.beta, 2)
    if nd.debt_to_equity_annual:
        valid = [x for x in nd.debt_to_equity_annual if x is not None and x == x]  # exclude NaN
        if valid:
            risk["debt_to_equity"] = _r(valid[-1], 2)
    if nd.short_pct_float is not None:
        risk["short_interest_pct_float"] = _r(nd.short_pct_float * 100, 1)
    if risk:
        metrics["risk"] = risk

    return metrics


# ---------------------------------------------------------------------------
# Input / output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AIInput:
    """Structured JSON snapshot sent to the AI verifier."""
    symbol: str
    as_of: str
    company_type: str
    data_quality: str
    block_scores: dict[str, float]
    horizon_scores: dict[str, float]
    critical_stop_factors: list[str]  # names of critical-severity stops only
    top_positive_features: list[str]
    top_negative_features: list[str]
    # All stop factors with descriptions (so AI understands WHY scores are affected)
    stop_factors: list[dict] = field(default_factory=list)
    # Trade recommendation fields (populated when available)
    trade_action: str = ""             # "Accumulate" | "Accumulate on Pullback" | "Avoid"
    current_price: Optional[float] = None
    limit_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    horizon_label: str = ""
    # Raw financial data for independent cross-check (populated when nd is available)
    raw_metrics: dict = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: AnalysisResult) -> AIInput:
        """Build AIInput from a completed AnalysisResult."""
        all_metrics: list[tuple[str, float]] = []
        for block_name, block_score in result.block_scores.items():
            for metric, score in block_score.breakdown.items():
                all_metrics.append((f"{block_name}.{metric}", score))

        all_metrics.sort(key=lambda x: x[1])
        top_negative = [f"{name}={score:.1f}" for name, score in all_metrics[:3]]
        top_positive = [f"{name}={score:.1f}" for name, score in reversed(all_metrics[-3:])]

        critical_stops = [
            sf.name for sf in result.stop_factors if sf.severity == "critical"
        ]
        all_stops = [
            {"name": sf.name, "severity": sf.severity, "description": sf.description}
            for sf in result.stop_factors
        ]

        rec = result.trade_rec
        raw_metrics = _extract_raw_metrics(result.nd) if result.nd is not None else {}

        return cls(
            symbol=result.ticker,
            as_of=date.today().isoformat(),
            company_type=result.company_type.value,
            data_quality=result.data_confidence,
            block_scores={k: round(v.score, 2) for k, v in result.block_scores.items()},
            horizon_scores={
                "short":  round(result.horizon.short, 1),
                "medium": round(result.horizon.medium, 1),
                "long":   round(result.horizon.long, 1),
            },
            critical_stop_factors=critical_stops,
            stop_factors=all_stops,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            trade_action=rec.action if rec else "",
            current_price=result.current_price,
            limit_price=rec.limit_price if rec else None,
            target_price=rec.target_price if rec else None,
            stop_price=rec.stop_price if rec else None,
            horizon_label=rec.horizon_label if rec else "",
            raw_metrics=raw_metrics,
        )

    def to_dict(self) -> dict:
        d: dict = {
            "symbol":                self.symbol,
            "as_of":                 self.as_of,
            "company_type":          self.company_type,
            "data_quality":          self.data_quality,
            "block_scores":          self.block_scores,
            "horizon_scores":        self.horizon_scores,
            "critical_stop_factors": self.critical_stop_factors,
            "top_positive_features": self.top_positive_features,
            "top_negative_features": self.top_negative_features,
        }
        if self.stop_factors:
            d["stop_factors"] = self.stop_factors
        if self.raw_metrics:
            d["raw_metrics"] = self.raw_metrics
        if self.trade_action:
            d["trade_action"] = self.trade_action
        if self.current_price is not None:
            d["current_price"] = round(self.current_price, 2)
        if self.limit_price is not None:
            d["limit_price"] = round(self.limit_price, 2)
        if self.target_price is not None:
            d["target_price"] = round(self.target_price, 2)
        if self.stop_price is not None:
            d["stop_price"] = round(self.stop_price, 2)
        if self.horizon_label:
            d["horizon_label"] = self.horizon_label
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class AIReview:
    """Structured response from the AI verifier."""
    agreement:      bool
    contradictions: list[str]
    confidence:     float        # 0.0 – 1.0
    narrative:      str
    action:         str          # "buy" | "watch" | "hold" | "avoid"
    abstain:        bool
    backend:        str = "unknown"
    input_tokens:   int = 0      # prompt tokens used
    output_tokens:  int = 0      # completion tokens used

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.action = str(self.action).lower()
        if self.action not in {"buy", "watch", "hold", "avoid", "accumulate", "accumulate_on_pullback"}:
            self.action = "hold"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# AI call logger
# ---------------------------------------------------------------------------

# Haiku pricing (per million tokens, USD) — update if model changes
_COST_PER_M = {
    "claude-haiku-4-5":         {"input": 0.80,  "output": 4.00},
    "claude-3-5-haiku-20241022": {"input": 0.80,  "output": 4.00},
    "claude-opus-4-5":          {"input": 15.00, "output": 75.00},
}


def _write_ai_log(
    symbol: str,
    model: str,
    system_prompt: str,
    user_payload: str,
    raw_response: str,
    input_tokens: int,
    output_tokens: int,
    reports_dir: str | Path = "reports",
) -> Path:
    """Append a full AI call record to reports/<SYMBOL>_ai_debug.txt.

    In production mode (STOCK_ANALYST_ENV=production) skips file writing
    to prevent unbounded disk accumulation inside the container.
    """
    import os
    production = os.environ.get("STOCK_ANALYST_ENV", "dev").lower() == "production"

    out_dir  = Path(reports_dir)
    log_path = out_dir / f"{symbol}_ai_debug.txt"

    pricing = _COST_PER_M.get(model, {"input": 0.0, "output": 0.0})
    cost_usd = (
        input_tokens  / 1_000_000 * pricing["input"]
        + output_tokens / 1_000_000 * pricing["output"]
    )
    total_tokens = input_tokens + output_tokens

    sep  = "═" * 70
    sep2 = "─" * 70
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    block = (
        f"\n{sep}\n"
        f"  AI CALL  —  {symbol}  |  {model}  |  {ts}\n"
        f"{sep}\n"
        f"\n── TOKENS ──────────────────────────────────────────────────────────\n"
        f"  Input  : {input_tokens:>6} tokens\n"
        f"  Output : {output_tokens:>6} tokens\n"
        f"  Total  : {total_tokens:>6} tokens\n"
        f"  Cost   : ${cost_usd:.6f} USD  (model: {model})\n"
        f"\n── SYSTEM PROMPT ────────────────────────────────────────────────────\n"
        f"{system_prompt}\n"
        f"\n── USER PAYLOAD (data sent to AI) ───────────────────────────────────\n"
        f"{user_payload}\n"
        f"\n── RAW AI RESPONSE ──────────────────────────────────────────────────\n"
        f"{raw_response}\n"
        f"\n{sep2}\n"
    )

    if production:
        # In production just print token cost to stdout — no disk writes.
        print(f"  AI [{symbol}] {input_tokens}+{output_tokens} tokens  ${cost_usd:.6f}")
        return log_path

    out_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(block)

    return log_path


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a second-opinion analyst verifying a quantitative stock analysis.
The analysis is produced by a rule-based scoring engine. Your job is BOTH to:
  1. Check internal consistency of the engine's scores and recommendation.
  2. Cross-check those scores against the raw financial data provided in
     "raw_metrics" — and flag cases where a score seems implausible given
     the underlying numbers.

━━ SCORING SCALE ━━
- Block scores (quality, valuation, technical, risk, style_fit): 0–10.
  ≥ 8 = strong  |  5–8 = acceptable  |  < 5 = weak
- Horizon scores (short, medium, long): 0–100 overall weighted scores.
- Block weights differ by horizon:
    short:  technical 40%, risk 25%, valuation 20%, quality 10%, style_fit 5%
    medium: valuation 30%, quality 25%, technical 20%, risk 15%, style_fit 10%
    long:   quality 35%, risk 25%, style_fit 15%, valuation 15%, technical 10%
  → A weak technical score on a long-horizon trade is expected and NOT a contradiction.

━━ HOW TO USE raw_metrics ━━
Use these real financial numbers to sanity-check whether the block scores make sense.
Examples of contradictions to flag:
  - valuation score ≥ 8 but pe_trailing > 40 AND revenue_growth_1y_pct < 10%
    (expensive stock at low growth ≠ good valuation)
  - quality score ≥ 8 but net_margin_pct < 5% or roe_pct < 8%
  - quality score ≥ 8 but revenue_growth_1y_pct < 0 (shrinking revenue)
  - risk score ≥ 7 but debt_to_equity > 2.0 or beta > 2.0
  - technical score < 3 but price_vs_ma200_pct > 20% (wildly extended) — mention in narrative
  - analyst_consensus = "sell" or "strong_sell" while action is "Accumulate"
  - analyst_upside_pct < 0 (price above median target) while action is "Accumulate"
  Note: raw_metrics may be partially absent — only flag if the available data
  clearly contradicts the score. Do NOT penalise for missing fields.

━━ TRADE ACTION SEMANTICS ━━
- "Accumulate" (market): engine confirmed technicals AND fundamentals are aligned.
  Flag contradiction only if block scores clearly don't support immediate entry.
- "Accumulate on Pullback" (limit order): engine ALREADY detected an issue
  (weak technicals, extended price, or poor R/R at market) and downgraded.
  The limit_price IS the intended entry. Technical weakness is NOT a contradiction
  here — it is the reason for the limit. Evaluate R/R at limit_price instead.
- "Avoid": verify that critical stop factors or score < 60 justify this.

━━ R/R VERIFICATION (when limit_price, target_price, stop_price are provided) ━━
  R/R = (target_price − entry) / (entry − stop_price), where entry = limit_price
  if action is "Accumulate on Pullback", else current_price.
  R/R < 1.5 is a real contradiction worth flagging.

━━ WHEN TO ABSTAIN ━━
  abstain=true ONLY when: data_quality="poor", OR critical_stop_factors are
  present but action is not "Avoid", OR R/R < 1.0 (clearly unfavourable).
  Do NOT abstain just because one block is weak.

━━ OUTPUT ━━
Respond ONLY with valid JSON. No markdown. No text outside the JSON.
Write "narrative" in English (1–2 sentences max, for a Telegram signal post).
{
  "agreement": <boolean — no genuine contradictions found?>,
  "contradictions": [<short strings, empty list if none>],
  "confidence": <float 0.0–1.0 — your certainty in the verdict>,
  "narrative": <1-2 sentences in English for a Telegram signal post>,
  "action": <"accumulate" | "accumulate_on_pullback" | "avoid" | "watch">,
  "abstain": <boolean>
}"""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AIConnector(abc.ABC):
    """Abstract base for AI verifier backends."""

    @abc.abstractmethod
    def review(self, ai_input: AIInput) -> AIReview:
        """Run AI consistency check and return structured review."""

    def _parse_response(self, text: str, backend: str) -> AIReview:
        """Parse a JSON-formatted LLM response into AIReview.

        Handles markdown fences and extracts the first JSON object via regex.
        Falls back to an abstain response when parsing fails.
        """
        import re as _re
        stripped = text.strip()

        # Strategy 1: extract first {...} block (works even with surrounding text)
        match = _re.search(r"\{.*\}", stripped, _re.DOTALL)
        if match:
            stripped = match.group(0)
        elif stripped.startswith("```"):
            # Fallback: strip fence lines
            lines = [ln for ln in stripped.splitlines() if not ln.startswith("```")]
            stripped = "\n".join(lines).strip()

        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            return AIReview(
                agreement=False,
                contradictions=[f"Failed to parse AI response: {exc}"],
                confidence=0.0,
                narrative=f"AI returned unparseable response: {text[:200]}",
                action="hold",
                abstain=True,
                backend=backend,
            )

        return AIReview(
            agreement=bool(data.get("agreement", False)),
            contradictions=list(data.get("contradictions") or []),
            confidence=float(data.get("confidence", 0.5)),
            narrative=str(data.get("narrative", "")),
            action=str(data.get("action", "hold")),
            abstain=bool(data.get("abstain", False)),
            backend=backend,
        )


# ---------------------------------------------------------------------------
# Null backend (always abstains)
# ---------------------------------------------------------------------------

class NullConnector(AIConnector):
    """No-op connector that always abstains.

    Used as the default when no AI backend is configured.
    """

    def review(self, ai_input: AIInput) -> AIReview:
        return AIReview(
            agreement=False,
            contradictions=[],
            confidence=0.0,
            narrative="No AI connector configured.",
            action="hold",
            abstain=True,
            backend="null",
        )


# ---------------------------------------------------------------------------
# Claude (Anthropic) backend
# ---------------------------------------------------------------------------

class ClaudeConnector(AIConnector):
    """Anthropic Claude API backend.

    Requires the ``anthropic`` package::

        pip install anthropic
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 2048,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from exc
        self._client = _anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def review(self, ai_input: AIInput) -> AIReview:
        user_payload = ai_input.to_json()
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_payload}],
        )
        raw_text     = message.content[0].text
        input_tokens  = message.usage.input_tokens
        output_tokens = message.usage.output_tokens

        _write_ai_log(
            symbol=ai_input.symbol,
            model=self._model,
            system_prompt=_SYSTEM_PROMPT,
            user_payload=user_payload,
            raw_response=raw_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        review = self._parse_response(raw_text, backend=f"claude/{self._model}")
        review.input_tokens  = input_tokens
        review.output_tokens = output_tokens
        return review


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class OllamaConnector(AIConnector):
    """Local Ollama HTTP API backend.

    Ollama must be running at ``base_url`` (default: http://localhost:11434).
    No additional Python packages required.
    """

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def review(self, ai_input: AIInput) -> AIReview:
        import urllib.error
        import urllib.request

        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": ai_input.to_json()},
            ],
            "stream": False,
        }).encode()

        url = f"{self._base_url}/api/chat"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            raise ConnectionError(
                f"Ollama not available at {self._base_url}: {exc}"
            ) from exc

        text = data["message"]["content"]
        return self._parse_response(text, backend=f"ollama/{self._model}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_connector(backend: str, **kwargs: object) -> AIConnector:
    """Create an AIConnector by backend name.

    Parameters
    ----------
    backend : str
        One of ``"null"``, ``"claude"``, ``"ollama"``.
    **kwargs :
        Forwarded to the connector constructor.

        - ``claude``: ``api_key`` (required), ``model``, ``max_tokens``
        - ``ollama``: ``model``, ``base_url``, ``timeout``

    Examples
    --------
    >>> connector = build_connector("null")
    >>> connector = build_connector("claude", api_key="sk-ant-...")
    >>> connector = build_connector("ollama", model="mistral")
    """
    if backend == "null":
        return NullConnector()
    if backend == "claude":
        return ClaudeConnector(**kwargs)  # type: ignore[arg-type]
    if backend == "ollama":
        return OllamaConnector(**kwargs)  # type: ignore[arg-type]
    raise ValueError(
        f"Unknown backend {backend!r}. Choose from: null, claude, ollama"
    )
