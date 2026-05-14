"""
Tests for src/data/sec_edgar.py

All network calls are mocked — no internet required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.sec_edgar import CONCEPTS, _extract_concept, fetch_fundamentals, get_cik


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_TICKER_MAP = {"AAPL": "0000320193", "MSFT": "0000789019"}

# Minimal companyfacts blob for AAPL with Revenue and NetIncomeLoss
_FACTS_BLOB = {
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"end": "2022-09-24", "start": "2021-09-26", "val": 394328000000, "form": "10-K", "filed": "2022-10-28", "accn": "a1"},
                        {"end": "2023-09-30", "start": "2022-10-02", "val": 383285000000, "form": "10-K", "filed": "2023-11-03", "accn": "a2"},
                        {"end": "2023-07-01", "start": "2023-04-02", "val": 81797000000,  "form": "10-Q", "filed": "2023-08-04", "accn": "a3"},
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {"end": "2022-09-24", "start": "2021-09-26", "val": 99803000000, "form": "10-K", "filed": "2022-10-28", "accn": "b1"},
                        {"end": "2023-09-30", "start": "2022-10-02", "val": 96995000000, "form": "10-K", "filed": "2023-11-03", "accn": "b2"},
                    ]
                }
            },
        }
    }
}


# ---------------------------------------------------------------------------
# get_cik
# ---------------------------------------------------------------------------

@patch("src.data.sec_edgar._load_ticker_map", return_value=_TICKER_MAP)
def test_get_cik_known_ticker(mock_map):
    assert get_cik("AAPL") == "0000320193"


@patch("src.data.sec_edgar._load_ticker_map", return_value=_TICKER_MAP)
def test_get_cik_case_insensitive(mock_map):
    assert get_cik("aapl") == "0000320193"


@patch("src.data.sec_edgar._load_ticker_map", return_value=_TICKER_MAP)
def test_get_cik_unknown_raises(mock_map):
    with pytest.raises(ValueError, match="CIK not found"):
        get_cik("FAKE")


# ---------------------------------------------------------------------------
# _extract_concept
# ---------------------------------------------------------------------------

def test_extract_concept_returns_dataframe():
    df = _extract_concept(_FACTS_BLOB, "Revenues", preferred_unit="USD")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert set(df.columns) >= {"end", "val", "form"}


def test_extract_concept_only_10k_and_10q():
    df = _extract_concept(_FACTS_BLOB, "Revenues", preferred_unit="USD")
    assert df["form"].isin(["10-K", "10-Q"]).all()


def test_extract_concept_deduplicates_by_end_and_form():
    # Inject two 10-K rows with same end date but different filed dates
    blob = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"end": "2023-09-30", "val": 383285000000, "form": "10-K", "filed": "2023-11-03", "accn": "new"},
                            {"end": "2023-09-30", "val": 999000000000, "form": "10-K", "filed": "2023-10-01", "accn": "old"},
                        ]
                    }
                }
            }
        }
    }
    df = _extract_concept(blob, "Revenues")
    annual = df[df["form"] == "10-K"]
    assert len(annual) == 1
    # most recently filed value should be kept
    assert annual.iloc[0]["val"] == 383285000000


def test_extract_concept_missing_tag_returns_empty():
    df = _extract_concept(_FACTS_BLOB, "NonExistentTag")
    assert df.empty


def test_extract_concept_sorted_by_end_date():
    df = _extract_concept(_FACTS_BLOB, "Revenues")
    assert df["end"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# fetch_fundamentals (mocked network)
# ---------------------------------------------------------------------------

@patch("src.data.sec_edgar._load_ticker_map", return_value=_TICKER_MAP)
@patch("src.data.sec_edgar._fetch_raw_facts", return_value=_FACTS_BLOB)
def test_fetch_fundamentals_returns_dict(mock_facts, mock_map):
    result = fetch_fundamentals("AAPL")
    assert isinstance(result, dict)
    assert len(result) > 0


@patch("src.data.sec_edgar._load_ticker_map", return_value=_TICKER_MAP)
@patch("src.data.sec_edgar._fetch_raw_facts", return_value=_FACTS_BLOB)
def test_fetch_fundamentals_revenue_is_dataframe(mock_facts, mock_map):
    result = fetch_fundamentals("AAPL")
    # "revenue" tries Revenues first — present in our blob
    assert "revenue" in result
    df = result["revenue"]
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


@patch("src.data.sec_edgar._load_ticker_map", return_value=_TICKER_MAP)
@patch("src.data.sec_edgar._fetch_raw_facts", return_value=_FACTS_BLOB)
def test_fetch_fundamentals_values_are_positive(mock_facts, mock_map):
    result = fetch_fundamentals("AAPL")
    for metric, df in result.items():
        assert (df["val"] != 0).any(), f"{metric} has all-zero values"


@patch("src.data.sec_edgar._load_ticker_map", return_value=_TICKER_MAP)
@patch("src.data.sec_edgar._fetch_raw_facts", return_value=_FACTS_BLOB)
def test_fetch_fundamentals_unknown_ticker_raises(mock_facts, mock_map):
    with pytest.raises(ValueError):
        fetch_fundamentals("FAKE")
