"""
Tests for src/data/cache.py

Covers: save/load DataFrame, save/load JSON, is_fresh TTL logic.
No network calls — fully offline.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

import src.data.cache as cache_mod


@pytest.fixture(autouse=True)
def tmp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect all cache I/O to a throwaway temp directory."""
    monkeypatch.setattr(cache_mod, "_CACHE_DIR", tmp_path / "cache")


# ---------------------------------------------------------------------------
# DataFrame cache
# ---------------------------------------------------------------------------

def test_save_and_load_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    cache_mod.save_df("test_df", df)
    loaded = cache_mod.load_df("test_df")
    assert loaded is not None
    pd.testing.assert_frame_equal(df, loaded)


def test_load_df_returns_none_when_missing():
    result = cache_mod.load_df("nonexistent_key")
    assert result is None


# ---------------------------------------------------------------------------
# JSON cache
# ---------------------------------------------------------------------------

def test_save_and_load_json_dict():
    data = {"ticker": "AAPL", "values": [1, 2, 3], "nested": {"x": True}}
    cache_mod.save_json("test_json", data)
    loaded = cache_mod.load_json("test_json")
    assert loaded == data


def test_save_and_load_json_list():
    data = ["AAPL", "MSFT", "NVDA"]
    cache_mod.save_json("test_list", data)
    assert cache_mod.load_json("test_list") == data


def test_load_json_returns_none_when_missing():
    assert cache_mod.load_json("no_such_key") is None


# ---------------------------------------------------------------------------
# is_fresh TTL
# ---------------------------------------------------------------------------

def test_is_fresh_after_save():
    df = pd.DataFrame({"v": [1]})
    cache_mod.save_df("fresh_key", df)
    assert cache_mod.is_fresh("fresh_key", max_age_hours=1.0, suffix=".parquet")


def test_is_fresh_returns_false_when_missing():
    assert not cache_mod.is_fresh("ghost_key", max_age_hours=1.0, suffix=".parquet")


def test_is_fresh_returns_false_after_ttl_expires(monkeypatch: pytest.MonkeyPatch):
    df = pd.DataFrame({"v": [1]})
    cache_mod.save_df("old_key", df)
    # simulate the clock being 2 hours in the future (file appears old)
    real_time = time.time()
    monkeypatch.setattr(time, "time", lambda: real_time + 2 * 3600)
    assert not cache_mod.is_fresh("old_key", max_age_hours=1.0, suffix=".parquet")
