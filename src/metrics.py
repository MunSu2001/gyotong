"""Shared regression metrics. Single source of truth across all models."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(list(actual), list(predicted)))


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(list(actual), list(predicted))))


def mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Mean Absolute Percentage Error (%). Robust to zeros via masking."""
    actual_arr = np.asarray(list(actual), dtype=float)
    predicted_arr = np.asarray(list(predicted), dtype=float)
    mask = actual_arr != 0
    if not mask.any():
        return float("nan")
    return float(
        np.mean(np.abs((actual_arr[mask] - predicted_arr[mask]) / actual_arr[mask])) * 100
    )


def summarize(
    actual: Iterable[float],
    predicted: Iterable[float],
) -> dict[str, float]:
    """Return MAE / RMSE / MAPE as a single dict (for logging or storage)."""
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mape": mape(actual, predicted),
    }


def fold_summary(fold_metrics: list[dict[str, float]]) -> pd.DataFrame:
    """Aggregate per-fold metrics into a tidy DataFrame with mean row."""
    df = pd.DataFrame(fold_metrics)
    mean_row = df.drop(columns=["fold"], errors="ignore").mean().to_frame().T
    mean_row.insert(0, "fold", "mean")
    return pd.concat([df, mean_row], ignore_index=True)
