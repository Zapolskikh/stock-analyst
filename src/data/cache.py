"""
Simple TTL-based file cache for fetched market data.

DataFrames  → stored as Parquet   (data/cache/<key>.parquet)
Dicts/lists → stored as JSON      (data/cache/<key>.json)

Cache lives in  data/cache/  relative to the working directory.
"""
from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import pandas as pd

_CACHE_DIR = Path("data/cache")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _path(key: str, suffix: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:48]
    return _CACHE_DIR / f"{safe}_{h}{suffix}"


def is_fresh(key: str, max_age_hours: float = 24.0, suffix: str = ".parquet") -> bool:
    """Return True if the cached file exists and is younger than *max_age_hours*."""
    p = _path(key, suffix)
    if not p.exists():
        return False
    return (time.time() - p.stat().st_mtime) < max_age_hours * 3600


# ---------------------------------------------------------------------------
# DataFrame cache
# ---------------------------------------------------------------------------

def save_df(key: str, df: pd.DataFrame) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_path(key, ".parquet"), index=True)


def load_df(key: str) -> pd.DataFrame | None:
    p = _path(key, ".parquet")
    return pd.read_parquet(p) if p.exists() else None


# ---------------------------------------------------------------------------
# JSON cache
# ---------------------------------------------------------------------------

def save_json(key: str, data: object) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_path(key, ".json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def load_json(key: str) -> dict | list | None:
    p = _path(key, ".json")
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)
