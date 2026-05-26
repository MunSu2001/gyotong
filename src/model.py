"""Part 3-A: Vacancy regression models.

Supports CatBoost (primary) and LightGBM (comparison) under a unified interface
via the abstract `VacancyModel` base class. This enables apples-to-apples
benchmarking and ensembling without code duplication.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from . import config
from .logging_setup import get_logger
from .metrics import fold_summary, summarize

# Matplotlib must be imported AFTER config sets MPLCONFIGDIR
config.ensure_directories()
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402


log = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# Unified model interface
# ─────────────────────────────────────────────────────────
class VacancyModel(ABC):
    """Abstract base class for vacancy-rate regression models."""

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> "VacancyModel":
        """Fit the model. Validation set is optional but used for early stopping."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict vacancy rates."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model to disk."""

    @abstractmethod
    def feature_importance(self) -> pd.Series:
        """Return feature importance keyed by feature name."""


# ─────────────────────────────────────────────────────────
# CatBoost implementation
# ─────────────────────────────────────────────────────────
class CatBoostVacancyModel(VacancyModel):
    """CatBoost regressor — handles categorical features natively."""

    name = "catboost"

    def __init__(self, cfg: config.CatBoostConfig = config.CatBoostConfig()) -> None:
        from catboost import CatBoostRegressor

        self._cfg = cfg
        self._model = CatBoostRegressor(
            iterations=cfg.iterations,
            learning_rate=cfg.learning_rate,
            depth=cfg.depth,
            eval_metric=cfg.eval_metric,
            early_stopping_rounds=cfg.early_stopping_rounds,
            random_seed=cfg.random_seed,
            verbose=cfg.verbose,
        )

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        cat_indices = [X_train.columns.get_loc(c) for c in config.CATEGORICAL_FEATURES]
        eval_set = (X_valid, y_valid) if X_valid is not None and y_valid is not None else None
        self._model.fit(
            X_train,
            y_train,
            cat_features=cat_indices,
            eval_set=eval_set,
            use_best_model=eval_set is not None,
        )
        return self

    def predict(self, X):
        return pd.Series(self._model.predict(X), index=X.index, name="predicted_vacancy")

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(path)

    def feature_importance(self):
        return pd.Series(
            self._model.feature_importances_,
            index=self._model.feature_names_,
            name="importance",
        ).sort_values(ascending=False)

    @property
    def raw_model(self):
        """Underlying CatBoost regressor (used by SHAP)."""
        return self._model


# ─────────────────────────────────────────────────────────
# LightGBM implementation
# ─────────────────────────────────────────────────────────
class LightGBMVacancyModel(VacancyModel):
    """LightGBM regressor — comparison baseline against CatBoost."""

    name = "lightgbm"

    def __init__(self, cfg: config.LightGBMConfig = config.LightGBMConfig()) -> None:
        import lightgbm as lgb

        self._cfg = cfg
        self._lgb = lgb
        self._model: Any | None = None

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        # LightGBM requires categorical features to be category dtype
        X_train_local = X_train.copy()
        for col in config.CATEGORICAL_FEATURES:
            X_train_local[col] = X_train_local[col].astype("category")

        callbacks = []
        eval_set = None
        if X_valid is not None and y_valid is not None:
            X_valid_local = X_valid.copy()
            for col in config.CATEGORICAL_FEATURES:
                X_valid_local[col] = X_valid_local[col].astype("category")
            eval_set = [(X_valid_local, y_valid)]
            callbacks = [self._lgb.early_stopping(self._cfg.early_stopping_rounds, verbose=False)]

        self._model = self._lgb.LGBMRegressor(
            n_estimators=self._cfg.n_estimators,
            learning_rate=self._cfg.learning_rate,
            max_depth=self._cfg.max_depth,
            num_leaves=self._cfg.num_leaves,
            random_state=self._cfg.random_state,
            verbose=self._cfg.verbose,
        )
        self._model.fit(
            X_train_local,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
            categorical_feature=list(config.CATEGORICAL_FEATURES),
        )
        return self

    def predict(self, X):
        X_local = X.copy()
        for col in config.CATEGORICAL_FEATURES:
            X_local[col] = X_local[col].astype("category")
        return pd.Series(
            self._model.predict(X_local), index=X.index, name="predicted_vacancy"
        )

    def save(self, path):
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)

    def feature_importance(self):
        return pd.Series(
            self._model.feature_importances_,
            index=self._model.feature_name_,
            name="importance",
        ).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────
# Walk-forward cross-validation
# ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CVResult:
    """Holds CV outputs for a single model."""

    model_name: str
    fold_metrics: pd.DataFrame
    final_model: VacancyModel


def cross_validate(
    df: pd.DataFrame,
    model_factory,
    cv_cfg: config.CrossValidationConfig = config.CrossValidationConfig(),
) -> CVResult:
    """Run time-series CV using the provided model factory (callable -> VacancyModel)."""
    feature_cols = list(config.FEATURE_COLS)
    X, y = df[feature_cols], df[config.TARGET_COL]

    tscv = TimeSeriesSplit(n_splits=cv_cfg.n_splits)
    fold_metrics = []

    for fold, (train_idx, valid_idx) in enumerate(tscv.split(X), start=1):
        model = model_factory()
        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
            X.iloc[valid_idx],
            y.iloc[valid_idx],
        )
        preds = model.predict(X.iloc[valid_idx])
        metrics = summarize(y.iloc[valid_idx], preds)
        metrics["fold"] = fold
        fold_metrics.append(metrics)
        log.info(
            "%s fold %d: MAE=%.4f MAPE=%.2f%% RMSE=%.4f",
            model.name, fold, metrics["mae"], metrics["mape"], metrics["rmse"],
        )

    metrics_df = fold_summary(fold_metrics)

    # Train final model on full data
    final_model = model_factory()
    final_model.fit(X, y)
    return CVResult(model_name=final_model.name, fold_metrics=metrics_df, final_model=final_model)


# ─────────────────────────────────────────────────────────
# Diagnostics (predictions + SHAP)
# ─────────────────────────────────────────────────────────
def save_diagnostics(
    result: CVResult,
    df: pd.DataFrame,
    output_dir: Path = config.IMG_DIR,
) -> pd.DataFrame:
    """Save scatter plot and SHAP summary for a trained model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    X = df[list(config.FEATURE_COLS)]
    y = df[config.TARGET_COL]
    y_pred = result.final_model.predict(X)

    pred_df = df[[config.DATE_COL, config.ROUTE_COL, config.TARGET_COL]].copy()
    pred_df["predicted_vacancy"] = y_pred.values

    # Scatter plot
    _save_scatter(y, y_pred, output_dir / f"{result.model_name}_pred_vs_actual.png")

    # SHAP (only for tree-based models with TreeExplainer support)
    if isinstance(result.final_model, CatBoostVacancyModel):
        _save_shap(result.final_model.raw_model, X, output_dir / f"{result.model_name}_shap.png")

    return pred_df


def _save_scatter(y_true: pd.Series, y_pred: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("Actual vacancy rate (%)")
    ax.set_ylabel("Predicted vacancy rate (%)")
    ax.set_title("Predicted vs Actual")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_shap(model: Any, X: pd.DataFrame, path: Path) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    X_display = X.rename(columns=config.FEATURE_DISPLAY_NAMES)
    shap.summary_plot(shap_values, X_display, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────
# Main: train both models, compare, save best
# ─────────────────────────────────────────────────────────
def main() -> None:
    from .io_utils import load_model_input

    df = load_model_input()
    log.info("data shape: %s", df.shape)

    results: dict[str, CVResult] = {}

    # CatBoost (primary)
    log.info("training CatBoost...")
    results["catboost"] = cross_validate(df, CatBoostVacancyModel)
    results["catboost"].final_model.save(config.CATBOOST_MODEL_FILE)
    save_diagnostics(results["catboost"], df)

    # LightGBM (comparison) — skip gracefully if not installed
    try:
        log.info("training LightGBM for comparison...")
        results["lightgbm"] = cross_validate(df, LightGBMVacancyModel)
        results["lightgbm"].final_model.save(config.LIGHTGBM_MODEL_FILE)
        save_diagnostics(results["lightgbm"], df)
    except ImportError:
        log.warning("LightGBM not installed, skipping comparison")

    # Comparative report
    log.info("=== Model comparison (mean across folds) ===")
    for name, res in results.items():
        mean_row = res.fold_metrics[res.fold_metrics["fold"] == "mean"].iloc[0]
        log.info(
            "%-10s MAE=%.4f MAPE=%.2f%% RMSE=%.4f",
            name, mean_row["mae"], mean_row["mape"], mean_row["rmse"],
        )


if __name__ == "__main__":
    main()
