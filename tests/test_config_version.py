"""Tests for src/models/config_version.py"""
from __future__ import annotations

import pytest

from src.models.config_version import (
    ConfigMeta,
    _CONFIG_REGISTRY,
    _CURRENT_VERSION,
    current_version,
    get_config_meta,
    list_versions,
)
from src.data.normalizer import NormalisedData
from src.engine.engine import analyse_nd


# ---------------------------------------------------------------------------
# Helper
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


# ---------------------------------------------------------------------------
# ConfigMeta dataclass
# ---------------------------------------------------------------------------

class TestConfigMeta:

    def test_all_fields_accessible(self):
        meta = ConfigMeta(
            version="1.0.0",
            valid_from="2025-01-01",
            trained_on_period="2018-2024",
            approved_by="test",
            reason_for_change="initial",
        )
        assert meta.version == "1.0.0"
        assert meta.valid_from == "2025-01-01"
        assert meta.trained_on_period == "2018-2024"
        assert meta.approved_by == "test"
        assert meta.reason_for_change == "initial"

    def test_frozen_cannot_mutate(self):
        meta = ConfigMeta("1.0.0", "2025-01-01", "2018-2024", "test", "init")
        with pytest.raises((AttributeError, TypeError)):
            meta.version = "2.0.0"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Registry functions
# ---------------------------------------------------------------------------

class TestGetConfigMeta:

    def test_default_returns_current(self):
        meta = get_config_meta()
        assert meta.version == current_version()

    def test_explicit_version_returned(self):
        meta = get_config_meta("1.0.0")
        assert meta.version == "1.0.0"

    def test_unknown_version_raises(self):
        with pytest.raises(KeyError):
            get_config_meta("99.99.99")

    def test_returns_config_meta_instance(self):
        meta = get_config_meta()
        assert isinstance(meta, ConfigMeta)

    def test_has_non_empty_fields(self):
        meta = get_config_meta()
        assert meta.version
        assert meta.valid_from
        assert meta.trained_on_period
        assert meta.approved_by
        assert meta.reason_for_change


class TestCurrentVersion:

    def test_returns_string(self):
        v = current_version()
        assert isinstance(v, str)

    def test_version_in_registry(self):
        v = current_version()
        assert v in _CONFIG_REGISTRY

    def test_matches_current_version_constant(self):
        assert current_version() == _CURRENT_VERSION


class TestListVersions:

    def test_returns_list(self):
        versions = list_versions()
        assert isinstance(versions, list)

    def test_current_version_included(self):
        assert current_version() in list_versions()

    def test_all_versions_in_registry(self):
        for v in list_versions():
            assert v in _CONFIG_REGISTRY


# ---------------------------------------------------------------------------
# Integration: AnalysisResult carries config_version
# ---------------------------------------------------------------------------

class TestConfigVersionInResult:

    def test_result_has_config_version(self):
        result = analyse_nd(_nd())
        assert hasattr(result, "config_version")

    def test_config_version_matches_current(self):
        result = analyse_nd(_nd())
        assert result.config_version == current_version()

    def test_config_version_is_non_empty(self):
        result = analyse_nd(_nd())
        assert result.config_version

    def test_config_version_in_report(self):
        from src.output.formatter import format_report
        result = analyse_nd(_nd())
        report = format_report(result)
        assert result.config_version in report

    def test_two_results_same_version(self):
        r1 = analyse_nd(_nd())
        r2 = analyse_nd(_nd(ticker="OTHER"))
        assert r1.config_version == r2.config_version
