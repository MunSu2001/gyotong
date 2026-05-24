"""Month-based walk-forward validation for the KTX vacancy model."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.model import CAT_FEATURES, DATA_PATH, FEATURE_COLS, TARGET_COL, load_model_input


OUTPUT_IMG_DIR = Path("outputs/img")
OUTPUT_DATA_DIR = Path("data/processed")
MONTH_METRICS_PATH = OUTPUT_DATA_DIR / "month_split_metrics.csv"
MONTH_PREDICTIONS_PATH = OUTPUT_DATA_DIR / "month_split_predictions.csv"
MONTH_COMPARISON_PATH = OUTPUT_DATA_DIR / "row_month_cv_comparison.csv"
MONTH_PLOT_PATH = OUTPUT_IMG_DIR / "month_pred_vs_actual.png"
COMPARISON_PLOT_PATH = OUTPUT_IMG_DIR / "row_month_cv_comparison.png"


def build_model(verbose: bool = False) -> CatBoostRegressor:
    """Return the baseline CatBoost regressor used for validation."""
    return CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        eval_metric="MAE",
        early_stopping_rounds=50,
        random_seed=42,
        verbose=verbose,
    )


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAPE while skipping zero denominators."""
    nonzero_mask = y_true != 0
    if not nonzero_mask.any():
        return float("nan")
    return (
        np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]))
        * 100
    )


def row_split_metrics(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Recompute the current row-based walk-forward metrics for comparison."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows: list[dict[str, float | int]] = []

    for fold, (train_idx, valid_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = build_model(verbose=False)
        model.fit(
            X_train,
            y_train,
            cat_features=CAT_FEATURES,
            eval_set=(X_valid, y_valid),
            use_best_model=True,
        )
        y_pred = model.predict(X_valid)

        fold_rows.append(
            {
                "fold": fold,
                "MAE": mean_absolute_error(y_valid, y_pred),
                "MAPE": safe_mape(y_valid.to_numpy(), y_pred),
                "RMSE": mean_squared_error(y_valid, y_pred) ** 0.5,
            }
        )

    return pd.DataFrame(fold_rows)


def month_split_validate(
    df: pd.DataFrame, n_splits: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run month-based walk-forward validation and collect fold predictions."""
    months = df["yearmonth"].drop_duplicates().sort_values().reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold, (train_month_idx, test_month_idx) in enumerate(tscv.split(months), start=1):
        train_months = months.iloc[train_month_idx]
        test_months = months.iloc[test_month_idx]
        overlap = len(set(train_months).intersection(set(test_months)))
        if overlap != 0:
            raise ValueError(f"Fold {fold} month overlap detected: {overlap}")

        train_df = df.loc[df["yearmonth"].isin(train_months)]
        test_df = df.loc[df["yearmonth"].isin(test_months)]

        X_train = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COL]
        X_test = test_df[FEATURE_COLS]
        y_test = test_df[TARGET_COL]

        model = build_model(verbose=False)
        model.fit(
            X_train,
            y_train,
            cat_features=CAT_FEATURES,
            eval_set=(X_test, y_test),
            use_best_model=True,
        )
        y_pred = model.predict(X_test)
        if len(y_pred) != len(test_df):
            raise ValueError(
                f"Fold {fold} prediction length mismatch: "
                f"{len(y_pred)} != {len(test_df)}"
            )

        fold_rows.append(
            {
                "fold": fold,
                "train_start": train_months.min(),
                "train_end": train_months.max(),
                "test_start": test_months.min(),
                "test_end": test_months.max(),
                "train_months": len(train_months),
                "test_months": len(test_months),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "MAE": mean_absolute_error(y_test, y_pred),
                "MAPE": safe_mape(y_test.to_numpy(), y_pred),
                "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
            }
        )

        prediction_frames.append(
            test_df[["yearmonth", "노선", TARGET_COL]]
            .assign(예측공실률=y_pred, fold=fold)
            .reset_index(drop=True)
        )

    metrics_df = pd.DataFrame(fold_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    return metrics_df, predictions_df


def make_comparison_df(row_metrics_df: pd.DataFrame, month_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact comparison table for row- and month-based validation."""
    row_mean = row_metrics_df[["MAE", "MAPE", "RMSE"]].mean()
    month_mean = month_metrics_df[["MAE", "MAPE", "RMSE"]].mean()

    comparison_df = pd.DataFrame(
        {
            "row_split_mean": row_mean,
            "month_split_mean": month_mean,
        }
    )
    comparison_df["difference(month-row)"] = (
        comparison_df["month_split_mean"] - comparison_df["row_split_mean"]
    )
    comparison_df.index.name = "metric"
    return comparison_df.reset_index()


def save_pred_plot(predictions_df: pd.DataFrame, output_path: Path = MONTH_PLOT_PATH) -> None:
    """Save month-based predicted-vs-actual scatter plot."""
    y_true = predictions_df[TARGET_COL]
    y_pred = predictions_df["예측공실률"]

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.65)
    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], "r--")
    plt.xlabel("Actual vacancy rate")
    plt.ylabel("Predicted vacancy rate")
    plt.title("Month Split Predicted vs Actual Vacancy Rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_comparison_plot(comparison_df: pd.DataFrame, output_path: Path = COMPARISON_PLOT_PATH) -> None:
    """Save row- versus month-based metric comparison bar chart."""
    metrics = comparison_df["metric"].tolist()
    row_values = comparison_df["row_split_mean"].to_numpy()
    month_values = comparison_df["month_split_mean"].to_numpy()
    positions = np.arange(len(metrics))
    width = 0.34

    plt.figure(figsize=(8, 5))
    plt.bar(positions - width / 2, row_values, width=width, label="row split")
    plt.bar(positions + width / 2, month_values, width=width, label="month split")
    plt.xticks(positions, metrics)
    plt.ylabel("Score")
    plt.title("Row Split vs Month Split CV Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    """Run month-based validation and save its artifacts."""
    OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_model_input(DATA_PATH)
    row_metrics_df = row_split_metrics(df)
    month_metrics_df, month_predictions_df = month_split_validate(df)
    comparison_df = make_comparison_df(row_metrics_df, month_metrics_df)

    month_metrics_df.to_csv(MONTH_METRICS_PATH, index=False, encoding="utf-8-sig")
    month_predictions_df.to_csv(MONTH_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    comparison_df.to_csv(MONTH_COMPARISON_PATH, index=False, encoding="utf-8-sig")

    save_pred_plot(month_predictions_df)
    save_comparison_plot(comparison_df)

    print(f"saved csv: {MONTH_METRICS_PATH}")
    print(f"saved csv: {MONTH_PREDICTIONS_PATH}")
    print(f"saved csv: {MONTH_COMPARISON_PATH}")
    print(f"saved plot: {MONTH_PLOT_PATH}")
    print(f"saved plot: {COMPARISON_PLOT_PATH}")
    print("\nmonth split mean metrics")
    print(month_metrics_df[["MAE", "MAPE", "RMSE"]].mean().round(4).to_string())
    print("\nrow vs month comparison")
    print(comparison_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
