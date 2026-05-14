"""Configuration version registry — calibration provenance tracking.

Tracks *when* scoring thresholds and weights were calibrated, *on which data*,
and *who approved* the change.  This makes every ``AnalysisResult`` reproducible:
given the same ticker data and the same ``config_version``, the result is
deterministic.

Usage::

    from src.models.config_version import get_config_meta, ConfigMeta

    meta = get_config_meta()
    print(meta.version)          # "1.0.0"
    print(meta.trained_on_period) # "2018-2024"

Bumping the version
-------------------
When thresholds or weights change:
1. Add a new entry to ``_CONFIG_REGISTRY`` below.
2. Update ``_CURRENT_VERSION`` to point to the new entry.
3. Commit the change — the git diff provides the full audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigMeta:
    """Provenance record for a particular set of scoring thresholds and weights.

    Fields
    ------
    version : str
        Semantic version string (MAJOR.MINOR.PATCH).
    valid_from : str
        ISO date from which this config is considered active (YYYY-MM-DD).
    trained_on_period : str
        Human-readable description of the calibration data period,
        e.g. ``"2018-2024"`` or ``"2020-Q1 – 2024-Q4"``.
    approved_by : str
        Name or identifier of the person who approved this version.
    reason_for_change : str
        Short description of what changed relative to the previous version.
    """
    version:            str
    valid_from:         str
    trained_on_period:  str
    approved_by:        str
    reason_for_change:  str


# ---------------------------------------------------------------------------
# Registry  (add new entries here when thresholds change)
# ---------------------------------------------------------------------------

_CONFIG_REGISTRY: dict[str, ConfigMeta] = {
    "1.0.0": ConfigMeta(
        version="1.0.0",
        valid_from="2025-01-01",
        trained_on_period="2018-2024",
        approved_by="initial",
        reason_for_change=(
            "Initial calibration — type-specific thresholds and block weights "
            "derived from S&P 500 constituent fundamentals 2018-2024."
        ),
    ),
}

_CURRENT_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_config_meta(version: str | None = None) -> ConfigMeta:
    """Return the :class:`ConfigMeta` for *version* (or the current version).

    Raises
    ------
    KeyError
        If *version* is specified but not registered.
    """
    if version is None:
        return _CONFIG_REGISTRY[_CURRENT_VERSION]
    return _CONFIG_REGISTRY[version]


def current_version() -> str:
    """Return the version string of the currently active configuration."""
    return _CURRENT_VERSION


def list_versions() -> list[str]:
    """Return all registered version strings, newest first."""
    return sorted(_CONFIG_REGISTRY.keys(), reverse=True)
