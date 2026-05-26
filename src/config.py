"""Centralized configuration for paths, columns, and model hyperparameters.

All modules MUST import constants from here rather than redefining them.
This prevents drift between feature engineering, training, and inference.

`.env` 파일이 프로젝트 루트에 있으면 자동으로 로드됩니다.
샘플은 `.env.sample` 참조.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ─── .env 자동 로드 (옵션, 없으면 무시) ──────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    # python-dotenv 미설치 시 시스템 환경변수만 사용
    pass


# ─────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
IMG_DIR = OUTPUTS_DIR / "img"
REPORTS_DIR = OUTPUTS_DIR / "reports"


# Matplotlib config directory (must be writable in sandboxed envs)
MPLCONFIGDIR = ROOT_DIR / ".matplotlib"


def ensure_directories() -> None:
    """Create all required output directories. Idempotent."""
    for directory in (PROCESSED_DIR, MODELS_DIR, IMG_DIR, REPORTS_DIR, MPLCONFIGDIR):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


# ─────────────────────────────────────────────────────────
# Data file paths
# ─────────────────────────────────────────────────────────
RAW_PASSENGER_XLSX = RAW_DIR / "high_speed_rail_passenger_trends.xlsx"
WIDE_CSV = PROCESSED_DIR / "ktx_wide.csv"
LONG_CSV = PROCESSED_DIR / "ktx_long.csv"
MODEL_INPUT_CSV = PROCESSED_DIR / "ktx_model_input.csv"

CATBOOST_MODEL_FILE = MODELS_DIR / "catboost_model.cbm"
LIGHTGBM_MODEL_FILE = MODELS_DIR / "lightgbm_model.pkl"

PROPHET_FORECAST_CSV = PROCESSED_DIR / "part4_prophet_forecast.csv"
PROPHET_COMPARISON_CSV = PROCESSED_DIR / "part4_prophet_vacancy_comparison.csv"
PROPHET_METRICS_CSV = PROCESSED_DIR / "part4_prophet_metrics.csv"

PART1_AGGREGATE_CSV = PROCESSED_DIR / "part1_route_aggregate.csv"
PART5_SHAP_CSV = PROCESSED_DIR / "part5_shap_importance.csv"
PART6_REPORT_JSON = REPORTS_DIR / "part6_llm_report.json"


# ─────────────────────────────────────────────────────────
# Column names
# ─────────────────────────────────────────────────────────
DATE_COL = "yearmonth"
ROUTE_COL = "노선"
TARGET_COL = "공실률"

FEATURE_COLS: tuple[str, ...] = (
    "월",
    "분기",
    "계절",
    "코로나기간",
    "SRT개통후",
    "노선",
    "공실률_lag1",
    "공실률_lag12",
    "공실률_ma3",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("노선", "계절")
LAG_FEATURES: tuple[str, ...] = ("공실률_lag1", "공실률_lag12", "공실률_ma3")

# SHAP/LLM 표시용 영문 매핑 (한 곳에서만 관리)
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "월": "month",
    "분기": "quarter",
    "계절": "season",
    "코로나기간": "covid_period",
    "SRT개통후": "after_srt",
    "노선": "route",
    "공실률_lag1": "vacancy_lag1",
    "공실률_lag12": "vacancy_lag12",
    "공실률_ma3": "vacancy_ma3",
}


# ─────────────────────────────────────────────────────────
# Routes & schedules
# ─────────────────────────────────────────────────────────
ROUTE_MAPPING: dict[str, tuple[str, str, str]] = {
    "경부선": ("경부선 KTX여객수", "경부선 KTX이용률", "경부선_KTX_공실률"),
    "호남선": ("호남선 KTX여객수", "호남선 KTX이용률", "호남선_KTX_공실률"),
    "경전선": ("경전선 KTX여객수", "경전선  KTX이용률", "경전선_KTX_공실률"),
    "전라선": ("전라선 KTX여객수", "전라선 KTX이용률", "전라선_KTX_공실률"),
    "동해선": ("동해선 KTX여객수", "동해선 KTX이용률", "동해선_KTX_공실률"),
}

COVID_START = "2020-01"
COVID_END = "2022-06"
SRT_OPEN_DATE = "2016-12"


# ─────────────────────────────────────────────────────────
# Model hyperparameters
# ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CatBoostConfig:
    iterations: int = 500
    learning_rate: float = 0.05
    depth: int = 6
    eval_metric: str = "MAE"
    early_stopping_rounds: int = 50
    random_seed: int = 42
    verbose: int = 100


@dataclass(frozen=True)
class LightGBMConfig:
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    num_leaves: int = 31
    early_stopping_rounds: int = 50
    random_state: int = 42
    verbose: int = -1


@dataclass(frozen=True)
class ProphetConfig:
    target_col: str = TARGET_COL
    horizon_months: int = 12
    test_months: int = 12
    yearly_seasonality: bool = True
    weekly_seasonality: bool = False
    daily_seasonality: bool = False
    interval_width: float = 0.8


@dataclass(frozen=True)
class CrossValidationConfig:
    n_splits: int = 5


# ─────────────────────────────────────────────────────────
# LLM (Part 6) configuration
# ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LLMConfig:
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o"))
    max_tokens: int = 2000
    temperature: float = 0.3
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: int = 30
