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
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.engine.engine import AnalysisResult


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
    critical_stop_factors: list[str]
    top_positive_features: list[str]
    top_negative_features: list[str]

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
            top_positive_features=top_positive,
            top_negative_features=top_negative,
        )

    def to_dict(self) -> dict:
        return {
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

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.action = str(self.action).lower()
        if self.action not in {"buy", "watch", "hold", "avoid"}:
            self.action = "hold"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a financial analysis consistency-checker. Your role is NOT to make
investment decisions but to verify the internal consistency of a quantitative
stock analysis produced by a rule-based scoring engine.

Given a JSON snapshot of the analysis, check:
1. Whether block scores are internally consistent.
2. Whether horizon scores make sense relative to each other.
3. Whether critical stop factors are reflected in the recommended action.
4. Whether data_quality warrants abstaining (abstain=true when data_quality=poor).

Respond ONLY with valid JSON (no markdown, no commentary outside the JSON object)
in this exact schema:
{
  "agreement": <boolean — is the analysis internally consistent?>,
  "contradictions": [<list of short contradiction strings, can be empty>],
  "confidence": <float 0.0–1.0>,
  "narrative": <single-paragraph human-readable summary>,
  "action": <"buy" | "watch" | "hold" | "avoid">,
  "abstain": <boolean — true if data_quality=poor or contradictions too severe>
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

        Strips markdown code fences if present.
        Falls back to an abstain response when parsing fails.
        """
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = [l for l in stripped.splitlines() if not l.startswith("```")]
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
        model: str = "claude-opus-4-5",
        max_tokens: int = 512,
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
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": ai_input.to_json()}],
        )
        text = message.content[0].text
        return self._parse_response(text, backend=f"claude/{self._model}")


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
        import urllib.request
        import urllib.error

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
