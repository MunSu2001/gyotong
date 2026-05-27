"""FastAPI application — serves analysis results and LLM insights to the dashboard.

Endpoints:
  GET  /health                          → liveness probe
  GET  /api/v1/aggregate                → Part 1 route aggregate
  GET  /api/v1/forecast                 → Part 4 Prophet forecast
  GET  /api/v1/shap                     → Part 5 feature importance
  GET  /api/v1/report                   → Part 6 cached LLM report
  POST /api/v1/report/refresh           → Force LLM regeneration
  POST /api/v1/report/regenerate        → Alias for /refresh
  GET  /api/v1/report/download/pdf      → Download business_report.pdf
  GET  /api/v1/report/download/md       → Download technical_report.md

Design principles:
- Lazy data loading with file-mtime cache invalidation
- Graceful degradation: missing files return 404, never 500
- All response models defined via Pydantic for schema docs
- CORS configured for Next.js dashboard on localhost:3001
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config
from src.logging_setup import get_logger


log = get_logger(__name__)
config.ensure_directories()


app = FastAPI(
    title="VACANT — KTX Vacancy Analytics API",
    description="Part 1~6 analysis results served to the Next.js dashboard.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────
# Static file mounts
# dashboard_static.html이 /data/processed/*.csv 를 직접 fetch하므로
# data/ 와 src/ 디렉토리를 정적으로 서빙한다.
# ─────────────────────────────────────────────────────────
_DATA_DIR = config.ROOT_DIR / "data"
_SRC_DIR = config.ROOT_DIR / "src"

if _DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(_DATA_DIR)), name="data")


@app.get("/dashboard", include_in_schema=False)
def serve_dashboard() -> FileResponse:
    """Serve dashboard_static.html at /dashboard."""
    html_path = _SRC_DIR / "dashboard_static.html"
    if not html_path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="dashboard_static.html not found")
    return FileResponse(str(html_path), media_type="text/html")


# ─────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: str


class InsightCardSchema(BaseModel):
    tab_id: str
    title: str
    body: str
    severity: str


class ReportSchema(BaseModel):
    generated_at: str
    headline: str
    summary: str
    actions: list[str]
    cards: list[InsightCardSchema]


class AggregateRow(BaseModel):
    route: str = Field(..., alias=config.ROUTE_COL)
    mean_vacancy: float
    rows: int

    class Config:
        populate_by_name = True


# ─────────────────────────────────────────────────────────
# Simple file-mtime cache
# ─────────────────────────────────────────────────────────
class _FileCache:
    """Cache file contents until mtime changes. Avoids repeated CSV reads."""

    def __init__(self) -> None:
        self._store: dict[Path, tuple[float, Any]] = {}

    def get(self, path: Path, loader):
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        cached = self._store.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        data = loader(path)
        self._store[path] = (mtime, data)
        return data


_cache = _FileCache()


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    from datetime import datetime, timezone
    return HealthResponse(status="ok", timestamp=datetime.now(timezone.utc).isoformat())


# ─────────────────────────────────────────────────────────
# Part 1: Route aggregate
# ─────────────────────────────────────────────────────────
@app.get("/api/v1/aggregate")
def get_aggregate() -> list[dict]:
    """Return per-route vacancy aggregate stats."""
    df = _cache.get(config.PART1_AGGREGATE_CSV, _load_csv)
    if df is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="Part 1 aggregate not generated yet. Run: python -m src.part1_aggregate",
        )
    return df.to_dict(orient="records")


# ─────────────────────────────────────────────────────────
# Part 4: Prophet forecast
# ─────────────────────────────────────────────────────────
@app.get("/api/v1/forecast")
def get_forecast(route: str | None = None) -> list[dict]:
    """Return Prophet forecast. Optionally filter by route name."""
    df = _cache.get(config.PROPHET_FORECAST_CSV, _load_csv)
    if df is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="Prophet forecast not generated yet. Run: python -m src.prophet_model",
        )
    if route:
        df = df[df[config.ROUTE_COL] == route]
        if df.empty:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, detail=f"no forecast for route: {route}"
            )
    return df.to_dict(orient="records")


# ─────────────────────────────────────────────────────────
# Part 5: SHAP importance
# ─────────────────────────────────────────────────────────
@app.get("/api/v1/shap")
def get_shap() -> list[dict]:
    """Return SHAP feature importance ranking."""
    df = _cache.get(config.PART5_SHAP_CSV, _load_csv)
    if df is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="SHAP importance not generated yet. Run: python -m src.shap_analysis",
        )
    return df.to_dict(orient="records")


# ─────────────────────────────────────────────────────────
# Part 6: LLM report (cached)
# ─────────────────────────────────────────────────────────
@app.get("/api/v1/report", response_model=ReportSchema)
def get_report() -> ReportSchema:
    """Return the most recent LLM-generated executive report."""
    data = _cache.get(config.PART6_REPORT_JSON, _load_json)
    if data is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="LLM report not generated yet. POST /api/v1/report/refresh to create.",
        )
    return ReportSchema.model_validate(data)


@app.post("/api/v1/report/refresh", response_model=ReportSchema)
def refresh_report() -> ReportSchema:
    """Trigger LLM regeneration. Useful when upstream data has changed."""
    from src.llm_report import build_snapshot_from_files, generate_report, save_report

    snapshot = build_snapshot_from_files()
    report = generate_report(snapshot)
    save_report(report)

    data = _load_json(config.PART6_REPORT_JSON)
    return ReportSchema.model_validate(data)


@app.post("/api/v1/report/regenerate", response_model=ReportSchema)
def regenerate_report() -> ReportSchema:
    """Alias for /report/refresh — regenerates LLM report and both downloadable files."""
    return refresh_report()


# ─────────────────────────────────────────────────────────
# Report file downloads
# ─────────────────────────────────────────────────────────
_BUSINESS_PDF = config.REPORTS_DIR / "business_report.pdf"
_TECHNICAL_PDF = config.REPORTS_DIR / "technical_report.pdf"


def _ensure_reports() -> None:
    """Generate report files on-demand if they don't exist yet."""
    if not _BUSINESS_PDF.exists() or not _TECHNICAL_PDF.exists():
        from src.report_generator import generate_business_pdf, generate_technical_pdf
        if not _BUSINESS_PDF.exists():
            generate_business_pdf(_BUSINESS_PDF)
        if not _TECHNICAL_PDF.exists():
            generate_technical_pdf(_TECHNICAL_PDF)


@app.get("/api/v1/report/download/pdf")
def download_pdf() -> FileResponse:
    """Download the business report as PDF (generated on first request if missing)."""
    try:
        _ensure_reports()
    except Exception as exc:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF 생성 실패: {exc}",
        )

    if not _BUSINESS_PDF.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="business_report.pdf not found. POST /api/v1/report/regenerate to create it.",
        )
    return FileResponse(
        path=str(_BUSINESS_PDF),
        media_type="application/pdf",
        filename="business_report.pdf",
    )


@app.get("/api/v1/report/download/md")
def download_technical_pdf() -> FileResponse:
    """Download the technical report as PDF (renamed from MD; URL kept for compat)."""
    try:
        _ensure_reports()
    except Exception as exc:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"기술 PDF 생성 실패: {exc}",
        )

    if not _TECHNICAL_PDF.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="technical_report.pdf not found. POST /api/v1/report/regenerate to create it.",
        )
    return FileResponse(
        path=str(_TECHNICAL_PDF),
        media_type="application/pdf",
        filename="technical_report.pdf",
    )