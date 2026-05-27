"""Shared IO utilities for loading and saving processed datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def load_model_input(path: Path = config.MODEL_INPUT_CSV) -> pd.DataFrame:
    """Load the canonical Part 3 model input.

    Sorted by (route, date) for deterministic downstream behavior.
    """
    df = pd.read_csv(path, parse_dates=[config.DATE_COL])
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df.sort_values([config.ROUTE_COL, config.DATE_COL]).reset_index(drop=True)


def save_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """Save a DataFrame as UTF-8-SIG CSV. Creates parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")


def load_csv(path: Path, *, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Convenience CSV loader with optional date parsing."""
    return pd.read_csv(path, parse_dates=parse_dates or [])
