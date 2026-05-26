"""Part 5: SHAP-based feature impact analysis on the trained CatBoost model.

Outputs:
  • CSV: ranked feature importance with mean(|SHAP|) values
  • PNG: SHAP summary plot (already saved by model.save_diagnostics, regenerated here for freshness)

Use cases:
  • Tab 4 of dashboard: shows which features drive vacancy predictions
  • Input to Part 6 LLM report: feeds importance ranking into prompt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .io_utils import load_model_input, save_csv
from .logging_setup import get_logger


log = get_logger(__name__)
config.ensure_directories()


def compute_shap_importance(model_path: Path = config.CATBOOST_MODEL_FILE) -> pd.DataFrame:
    """Compute mean absolute SHAP values per feature, sorted descending."""
    import shap
    from catboost import CatBoostRegressor

    if not model_path.exists():
        raise FileNotFoundError(
            f"trained model not found: {model_path}. Run: python -m src.model"
        )

    df = load_model_input()
    X = df[list(config.FEATURE_COLS)]

    model = CatBoostRegressor()
    model.load_model(model_path)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)

    importance = pd.DataFrame(
        {
            "feature": list(config.FEATURE_COLS),
            "feature_en": [
                config.FEATURE_DISPLAY_NAMES.get(f, f) for f in config.FEATURE_COLS
            ],
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance["rank"] = importance.index + 1
    return importance


def main() -> None:
    importance = compute_shap_importance()
    save_csv(importance, config.PART5_SHAP_CSV)
    log.info("saved SHAP importance: %s", config.PART5_SHAP_CSV)
    log.info("top features:\n%s", importance.head().to_string(index=False))


if __name__ == "__main__":
    main()
