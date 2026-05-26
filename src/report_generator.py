"""Part 6.5: Report generator — produces TWO distinct outputs.

  • business_report.pdf : 비즈니스 보고서 (운영자·경영진용)
                          기술 용어 없음. 인사이트 + 수치적 시각화 중심.
                          → reportlab + matplotlib (한글 폰트 안정 임베딩)

  • technical_report.md : 기술 리포트 (개발자·데이터사이언티스트용)
                          MAE/RMSE/MAPE, SHAP, CV fold, 재현 명령어 포함.

데이터 출처 (대시보드와 동일):
  - data/processed/ktx_long.csv                       (월별 노선 공실률)
  - data/processed/route_summary.csv                  (노선 메타)
  - data/processed/station_summary.csv                (역 승하차)
  - data/processed/model_comparison_per_route.csv     (회귀 모델 성능)
  - data/processed/shap_per_route.csv                 (변수 영향도)
  - data/processed/part4_prophet_forecast.csv         (시계열 예측)
  - data/processed/part4_prophet_metrics.csv          (시계열 모델 성능)
  - data/processed/part4_prophet_vacancy_comparison.csv (Prophet 검증)
  - data/processed/holiday_mannwhitney.csv            (명절 효과)
  - outputs/reports/part6_llm_report.json             (LLM 인사이트)
"""

from __future__ import annotations

import io
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config
from .logging_setup import get_logger


log = get_logger(__name__)
config.ensure_directories()


# ─────────────────────────────────────────────────────────
# Output paths
# ─────────────────────────────────────────────────────────
BUSINESS_PDF = config.REPORTS_DIR / "business_report.pdf"
TECHNICAL_MD = config.REPORTS_DIR / "technical_report.md"
PRETENDARD_DIR = config.ROOT_DIR / "assets" / "fonts"


# ─────────────────────────────────────────────────────────
# Data loaders (대시보드와 동일 파일들)
# ─────────────────────────────────────────────────────────
def _load_llm_report() -> dict | None:
    if not config.PART6_REPORT_JSON.exists():
        log.warning("LLM report not found: %s", config.PART6_REPORT_JSON)
        return None
    try:
        return json.loads(config.PART6_REPORT_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("failed to parse LLM report: %s", exc)
        return None


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log.warning("failed to read %s: %s", path, exc)
        return None


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None and not pd.isna(v) else default
    except (ValueError, TypeError):
        return default


def _parse_unified_mae(value) -> float | None:
    m = re.search(r"-?\d+(\.\d+)?", str(value or ""))
    return float(m.group(0)) if m else None


def _aggregate_route_vacancy(long_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """ktx_long.csv에서 노선별 평균 공실률 집계."""
    if long_df is None or long_df.empty:
        return None
    if config.ROUTE_COL not in long_df.columns or config.TARGET_COL not in long_df.columns:
        return None
    agg = (
        long_df.groupby(config.ROUTE_COL, as_index=False)
        .agg(mean_vacancy=(config.TARGET_COL, "mean"),
             min_vacancy=(config.TARGET_COL, "min"),
             max_vacancy=(config.TARGET_COL, "max"),
             rows=(config.TARGET_COL, "count"))
    )
    agg["mean_vacancy"] = agg["mean_vacancy"].round(2)
    return agg.sort_values("mean_vacancy", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────
# Matplotlib 한글 폰트 설정
# ─────────────────────────────────────────────────────────
def _ensure_pretendard_ttf() -> None:
    """Pretendard TTF가 없으면 OTF에서 자동 변환하여 생성.

    reportlab.TTFont는 OTF의 PostScript outlines를 지원하지 않으므로
    OTF만 있는 환경에서도 TTF를 즉석 생성한다 (cu2qu + fontTools).
    """
    PRETENDARD_DIR.mkdir(parents=True, exist_ok=True)
    needed = [
        (PRETENDARD_DIR / "Pretendard-Regular.otf", PRETENDARD_DIR / "Pretendard-Regular.ttf"),
        (PRETENDARD_DIR / "Pretendard-Bold.otf", PRETENDARD_DIR / "Pretendard-Bold.ttf"),
    ]
    todo = [(o, t) for o, t in needed if o.exists() and not t.exists()]
    if not todo:
        return
    try:
        from fontTools.ttLib import TTFont, newTable
        from fontTools.pens.ttGlyphPen import TTGlyphPen
        from cu2qu.pens import Cu2QuPen
    except ImportError as exc:
        log.warning("cu2qu/fontTools 미설치 — OTF→TTF 변환 불가: %s", exc)
        return

    def _convert(otf_path: Path, ttf_path: Path) -> None:
        font = TTFont(str(otf_path))
        if font.sfntVersion != "OTTO" or "CFF " not in font:
            return
        glyph_order = font.getGlyphOrder()
        font["loca"] = newTable("loca")
        font["glyf"] = glyf = newTable("glyf")
        glyf.glyphOrder = glyph_order
        glyph_set = font.getGlyphSet()
        glyphs = {}
        for gname in glyph_order:
            ttPen = TTGlyphPen(glyph_set)
            cu2quPen = Cu2QuPen(ttPen, 1.0, reverse_direction=True)
            glyph_set[gname].draw(cu2quPen)
            glyphs[gname] = ttPen.glyph()
        glyf.glyphs = glyphs
        del font["CFF "]
        glyf.compile(font)
        hmtx = font["hmtx"]
        for gname, g in glyf.glyphs.items():
            if hasattr(g, "xMin"):
                hmtx[gname] = (hmtx[gname][0], g.xMin)
        font["maxp"] = maxp = newTable("maxp")
        maxp.tableVersion = 0x00010000
        maxp.maxZones = 1
        maxp.maxTwilightPoints = 0
        maxp.maxStorage = 0
        maxp.maxFunctionDefs = 0
        maxp.maxInstructionDefs = 0
        maxp.maxStackElements = 0
        maxp.maxSizeOfInstructions = 0
        maxp.maxComponentElements = max(
            len(g.components if hasattr(g, "components") else []) for g in glyf.glyphs.values()
        )
        maxp.compile(font)
        post = font["post"]
        post.formatType = 2.0
        post.extraNames = []
        post.mapping = {}
        post.glyphOrder = glyph_order
        try:
            post.compile(font)
        except OverflowError:
            post.formatType = 3
        font.sfntVersion = "\x00\x01\x00\x00"
        font.save(str(ttf_path))
        log.info("converted OTF → TTF: %s", ttf_path.name)

    for otf, ttf in todo:
        try:
            _convert(otf, ttf)
        except Exception as exc:
            log.warning("OTF→TTF 변환 실패 %s: %s", otf.name, exc)


def _korean_font_paths() -> tuple[Path | None, Path | None]:
    """Pretendard TTF가 있으면 우선, 없으면 OTF. (regular, bold)."""
    reg_ttf = PRETENDARD_DIR / "Pretendard-Regular.ttf"
    bold_ttf = PRETENDARD_DIR / "Pretendard-Bold.ttf"
    reg_otf = PRETENDARD_DIR / "Pretendard-Regular.otf"
    bold_otf = PRETENDARD_DIR / "Pretendard-Bold.otf"
    reg = reg_ttf if reg_ttf.exists() else (reg_otf if reg_otf.exists() else None)
    bold = bold_ttf if bold_ttf.exists() else (bold_otf if bold_otf.exists() else reg)
    return reg, bold


def _setup_matplotlib() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    reg, bold = _korean_font_paths()
    if reg is not None:
        font_manager.fontManager.addfont(str(reg))
        if bold is not None and bold != reg:
            font_manager.fontManager.addfont(str(bold))
        plt.rcParams["font.family"] = "Pretendard"
    plt.rcParams["axes.unicode_minus"] = False


# ─────────────────────────────────────────────────────────
# Reportlab 한글 폰트 등록 (모듈 단위 1회)
# ─────────────────────────────────────────────────────────
_FONTS_REGISTERED = False


def _register_korean_fonts() -> tuple[str, str]:
    """Pretendard OTF를 reportlab에 등록. (regular_name, bold_name) 반환."""
    global _FONTS_REGISTERED
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    regular_name = "Pretendard"
    bold_name = "Pretendard-Bold"

    if _FONTS_REGISTERED:
        return regular_name, bold_name

    reg_path, bold_path = _korean_font_paths()

    if reg_path is None or not str(reg_path).endswith(".ttf"):
        # reportlab의 TTFont는 OTF의 PostScript outlines를 지원하지 않음.
        # OTF만 있다면 즉석에서 TTF로 변환을 시도한다.
        try:
            _ensure_pretendard_ttf()
            reg_path, bold_path = _korean_font_paths()
        except Exception as exc:
            log.warning("OTF→TTF 변환 실패: %s", exc)

    if reg_path is None or not str(reg_path).endswith(".ttf"):
        log.warning("Pretendard TTF 없음 — 기본 폰트 사용 (한글 깨질 수 있음)")
        return "Helvetica", "Helvetica-Bold"

    try:
        pdfmetrics.registerFont(TTFont(regular_name, str(reg_path)))
        if bold_path is not None and str(bold_path).endswith(".ttf"):
            pdfmetrics.registerFont(TTFont(bold_name, str(bold_path)))
        else:
            pdfmetrics.registerFont(TTFont(bold_name, str(reg_path)))
        from reportlab.pdfbase.pdfmetrics import registerFontFamily
        registerFontFamily(regular_name, normal=regular_name, bold=bold_name,
                           italic=regular_name, boldItalic=bold_name)
        _FONTS_REGISTERED = True
        log.info("registered Korean fonts: %s, %s", regular_name, bold_name)
    except Exception as exc:
        log.warning("failed to register Korean fonts: %s — fallback", exc)
        return "Helvetica", "Helvetica-Bold"

    return regular_name, bold_name


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers — matplotlib 이미지를 BytesIO로 반환
# ─────────────────────────────────────────────────────────────────────────────
def _chart_vacancy_by_route(agg: pd.DataFrame) -> io.BytesIO | None:
    """노선별 평균 공실률 가로 바 차트."""
    if agg is None or agg.empty:
        return None
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    sorted_agg = agg.sort_values("mean_vacancy", ascending=True)
    routes = sorted_agg[config.ROUTE_COL].tolist()
    vals = sorted_agg["mean_vacancy"].tolist()
    colors = [
        "#22c55e" if v < 15 else "#f59e0b" if v < 25 else "#ef4444"
        for v in vals
    ]
    bars = ax.barh(routes, vals, color=colors, height=0.55, edgecolor="white")

    # 음수값이 있으면 x축 범위에 좌측 패딩 확보 (라벨이 노선명과 겹치지 않게)
    v_min = min(vals)
    v_max = max(vals)
    has_neg = v_min < 0
    span = max(abs(v_min), v_max)
    x_min = v_min - span * 0.18 if has_neg else 0
    x_max = v_max + span * 0.18 if v_max > 0 else span * 0.18
    label_offset = (x_max - x_min) * 0.01

    if has_neg:
        ax.axvline(0, color="#94a3b8", linewidth=0.7)

    for bar, v in zip(bars, vals):
        if v >= 0:
            ax.text(v + label_offset, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", ha="left",
                    fontsize=10, fontweight="bold", color="#0f172a")
        else:
            # 음수: 막대 왼쪽 끝에 오른쪽 정렬로 배치 → y축 라벨과 분리
            ax.text(v - label_offset, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", ha="right",
                    fontsize=10, fontweight="bold", color="#0f172a")

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("평균 공실률 (%)", fontsize=10, color="#475569")
    ax.set_title("노선별 평균 공실률", fontsize=12, fontweight="bold", color="#0f172a", loc="left", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#475569")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _chart_vacancy_trend(long_df: pd.DataFrame | None) -> io.BytesIO | None:
    """노선별 연도 평균 공실률 추세선."""
    if long_df is None or long_df.empty:
        return None
    if config.ROUTE_COL not in long_df.columns or config.TARGET_COL not in long_df.columns:
        return None
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    df = long_df.copy()
    if "연도" not in df.columns and "yearmonth" in df.columns:
        df["연도"] = pd.to_datetime(df["yearmonth"], errors="coerce").dt.year
    if "연도" not in df.columns:
        return None

    grouped = (
        df.dropna(subset=["연도"])
        .groupby(["연도", config.ROUTE_COL], as_index=False)[config.TARGET_COL]
        .mean()
    )
    routes = grouped[config.ROUTE_COL].unique().tolist()
    if not routes:
        return None

    palette = ["#1d4ed8", "#16a34a", "#f59e0b", "#ef4444", "#8b5cf6", "#0ea5e9", "#d946ef"]
    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=150)
    for i, route in enumerate(sorted(routes)):
        rd = grouped[grouped[config.ROUTE_COL] == route].sort_values("연도")
        ax.plot(rd["연도"], rd[config.TARGET_COL],
                label=route, color=palette[i % len(palette)],
                linewidth=2.2, marker="o", markersize=4)

    ax.axhline(0, color="#94a3b8", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("연도", fontsize=10, color="#475569")
    ax.set_ylabel("평균 공실률 (%)", fontsize=10, color="#475569")
    ax.set_title("연도별 노선 공실률 추이", fontsize=12, fontweight="bold", color="#0f172a", loc="left", pad=12)
    ax.legend(loc="best", fontsize=8, frameon=False, ncol=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#475569")
    ax.set_axisbelow(True)
    ax.grid(color="#e2e8f0", linewidth=0.7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _save_chart_png(fig, filename: str) -> Path | None:
    """matplotlib Figure를 outputs/reports/ (마크다운과 동일 폴더)에 저장.

    이렇게 하면 마크다운에서 파일명만으로 참조 가능 → 모든 뷰어 호환.
    """
    import matplotlib.pyplot as plt
    out_path = config.REPORTS_DIR / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", bbox_inches="tight", facecolor="white", dpi=140)
    plt.close(fig)
    return out_path


def _chart_regression_per_route(model_comparison: pd.DataFrame | None) -> Path | None:
    """노선별 개별 모델 MAE/RMSE 가로 막대 차트 + 통합모델 기준선."""
    if model_comparison is None or model_comparison.empty:
        return None
    if "노선" not in model_comparison.columns or "개별모델_MAE" not in model_comparison.columns:
        return None
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    df = model_comparison.copy().sort_values("개별모델_MAE")
    routes = df["노선"].tolist()
    mae = df["개별모델_MAE"].astype(float).tolist()
    rmse = df["개별모델_RMSE"].astype(float).tolist() if "개별모델_RMSE" in df.columns else None
    unified = _parse_unified_mae(df.iloc[0].get("통합모델_MAE"))

    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=140)
    y = np.arange(len(routes))
    h = 0.38
    ax.barh(y - h / 2, mae, height=h, color="#1d4ed8", label="MAE")
    if rmse:
        ax.barh(y + h / 2, rmse, height=h, color="#94a3b8", label="RMSE")

    # 라벨 오프셋을 데이터 범위 기반으로 계산
    max_val = max(mae + (rmse or []) + ([unified] if unified else []))
    offset = max_val * 0.015
    for i, v in enumerate(mae):
        ax.text(v + offset, y[i] - h / 2, f"{v:.2f}",
                va="center", ha="left", fontsize=8.5, color="#0f172a")
    if rmse:
        for i, v in enumerate(rmse):
            ax.text(v + offset, y[i] + h / 2, f"{v:.2f}",
                    va="center", ha="left", fontsize=8.5, color="#475569")
    # 라벨까지 들어갈 공간 확보
    ax.set_xlim(0, max_val * 1.15)

    if unified is not None:
        ax.axvline(unified, color="#ef4444", linestyle="--", linewidth=1.4,
                   label=f"Unified MAE = {unified:.2f}")

    ax.set_yticks(y, routes)
    ax.set_xlabel("Error (% vacancy)", fontsize=10, color="#475569")
    ax.set_title("CatBoost — Per-route CV Error", fontsize=12, fontweight="bold",
                 color="#0f172a", loc="left", pad=10)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.7)
    fig.tight_layout()
    return _save_chart_png(fig, "metric_regression_per_route.png")


def _chart_prophet_per_route(prophet_df: pd.DataFrame | None) -> Path | None:
    """Prophet MAE/RMSE/bias 노선별 차트 (그룹 막대)."""
    if prophet_df is None or prophet_df.empty:
        return None
    if "노선" not in prophet_df.columns or "mae_pp" not in prophet_df.columns:
        return None
    _setup_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    df = prophet_df.copy().sort_values("mae_pp")
    routes = df["노선"].tolist()
    mae = df["mae_pp"].astype(float).tolist()
    rmse = df["rmse_pp"].astype(float).tolist() if "rmse_pp" in df.columns else None
    bias = df["mean_error_pp"].astype(float).tolist() if "mean_error_pp" in df.columns else None

    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=140)
    x = np.arange(len(routes))
    w = 0.27
    ax.bar(x - w, mae, width=w, color="#1d4ed8", label="MAE (pp)")
    if rmse:
        ax.bar(x, rmse, width=w, color="#94a3b8", label="RMSE (pp)")
    if bias:
        ax.bar(x + w, bias, width=w, color="#f59e0b", label="Bias (pp)")
    ax.axhline(0, color="#475569", linewidth=0.6)

    # y축 범위에 음수 패딩 확보 (라벨 + 범례 공간)
    all_vals = mae + (rmse or []) + (bias or [])
    y_min = min(all_vals + [0])
    y_max = max(all_vals + [0])
    span = y_max - y_min
    pad_top = span * 0.18  # 범례 + 라벨 공간
    pad_bot = span * 0.10 if y_min < 0 else 0
    ax.set_ylim(y_min - pad_bot, y_max + pad_top)
    label_off = span * 0.015

    def _label(xs, vals, color):
        for i, v in enumerate(vals):
            if v >= 0:
                ax.text(xs[i], v + label_off, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=8, color=color)
            else:
                ax.text(xs[i], v - label_off, f"{v:.1f}",
                        ha="center", va="top", fontsize=8, color=color)

    _label(x - w, mae, "#0f172a")
    if rmse:
        _label(x, rmse, "#475569")
    if bias:
        _label(x + w, bias, "#92400e")

    ax.set_xticks(x, routes)
    ax.set_ylabel("Error (percentage point)", fontsize=10, color="#475569")
    ax.set_title("Prophet — Per-route Holdout Error", fontsize=12, fontweight="bold",
                 color="#0f172a", loc="left", pad=10)
    ax.legend(loc="upper right", fontsize=9, frameon=False, ncol=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    fig.tight_layout()
    return _save_chart_png(fig, "metric_prophet_per_route.png")


def _chart_shap_importance(shap_df: pd.DataFrame | None) -> Path | None:
    """SHAP 변수별 평균 + std (errorbar) 가로 막대 차트."""
    if shap_df is None or shap_df.empty:
        return None
    if "변수" not in shap_df.columns or "SHAP중요도" not in shap_df.columns:
        return None
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    agg = (
        shap_df.groupby("변수", as_index=False)
        .agg(mean_abs_shap=("SHAP중요도", "mean"),
             std=("SHAP중요도", "std"))
        .sort_values("mean_abs_shap")
    )
    feats = agg["변수"].tolist()
    means = agg["mean_abs_shap"].tolist()
    stds = agg["std"].fillna(0).tolist()

    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=140)
    ax.barh(feats, means, xerr=stds, color="#16a34a", ecolor="#475569",
            capsize=3, error_kw={"linewidth": 1.0})
    max_extent = max((m + s) for m, s in zip(means, stds))
    label_off = max_extent * 0.02
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + label_off, i, f"{m:.3f}",
                va="center", ha="left", fontsize=8.5, color="#0f172a")
    ax.set_xlim(0, max_extent * 1.18)
    ax.set_xlabel("mean |SHAP|  (errorbar = std across routes)", fontsize=10, color="#475569")
    ax.set_title("Feature Importance — SHAP (CatBoost)", fontsize=12, fontweight="bold",
                 color="#0f172a", loc="left", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.7)
    fig.tight_layout()
    return _save_chart_png(fig, "metric_shap_importance.png")


def _chart_residual_distribution(comparison_df: pd.DataFrame | None) -> Path | None:
    """Prophet 잔차(error) 분포 — 히스토그램 + 노선별 박스플롯."""
    if comparison_df is None or comparison_df.empty or "error" not in comparison_df.columns:
        return None
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    errs = comparison_df["error"].astype(float).dropna().tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=140,
                                    gridspec_kw={"width_ratios": [1, 1.2]})
    # histogram
    ax1.hist(errs, bins=20, color="#1d4ed8", edgecolor="white", alpha=0.85)
    ax1.axvline(0, color="#475569", linewidth=1.0)
    mean_err = sum(errs) / len(errs)
    ax1.axvline(mean_err, color="#ef4444", linestyle="--", linewidth=1.4,
                label=f"mean = {mean_err:+.2f}")
    ax1.set_xlabel("error  (actual − predicted, pp)", fontsize=9.5, color="#475569")
    ax1.set_ylabel("count", fontsize=9.5, color="#475569")
    ax1.set_title("Residual Distribution", fontsize=11, fontweight="bold",
                  color="#0f172a", loc="left", pad=8)
    ax1.legend(fontsize=9, frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", color="#e2e8f0", linewidth=0.6)
    ax1.set_axisbelow(True)

    # boxplot per route
    if "노선" in comparison_df.columns:
        routes = sorted(comparison_df["노선"].unique().tolist())
        groups = [comparison_df.loc[comparison_df["노선"] == r, "error"].astype(float).tolist()
                  for r in routes]
        bp = ax2.boxplot(groups, labels=routes, patch_artist=True, widths=0.55,
                         medianprops={"color": "#0f172a", "linewidth": 1.4})
        for patch in bp["boxes"]:
            patch.set_facecolor("#dbeafe")
            patch.set_edgecolor("#1d4ed8")
        ax2.axhline(0, color="#475569", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("error (pp)", fontsize=9.5, color="#475569")
        ax2.set_title("Residual by Route", fontsize=11, fontweight="bold",
                      color="#0f172a", loc="left", pad=8)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(axis="y", color="#e2e8f0", linewidth=0.6)
        ax2.set_axisbelow(True)

    fig.tight_layout()
    return _save_chart_png(fig, "metric_residual_distribution.png")


def _md_chart_link(path: Path | None, alt: str) -> str:
    """PNG 경로를 상대 경로로 참조 — VS Code 등 로컬 뷰어에서 정상 표시."""
    if path is None or not path.exists():
        return ""
    try:
        rel_path = path.relative_to(TECHNICAL_MD.parent)
    except ValueError:
        rel_path = path

    return f"![{alt}]({rel_path.as_posix()})"


def _chart_monthly_pattern(long_df: pd.DataFrame | None) -> io.BytesIO | None:
    """월별 평균 공실률 바 차트."""
    if long_df is None or long_df.empty or "월" not in long_df.columns:
        return None
    if config.TARGET_COL not in long_df.columns:
        return None
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    by_month = (
        long_df.groupby("월", as_index=False)[config.TARGET_COL]
        .mean()
        .sort_values("월")
    )
    if by_month.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.4, 3.0), dpi=150)
    months = by_month["월"].astype(int).tolist()
    vals = by_month[config.TARGET_COL].tolist()
    avg = sum(vals) / len(vals)
    colors = ["#1d4ed8" if v < avg else "#f59e0b" for v in vals]
    bars = ax.bar([f"{m}월" for m in months], vals, color=colors, edgecolor="white", width=0.7)

    # y축 범위 + 라벨 오프셋
    v_min = min(vals + [0])
    v_max = max(vals + [0])
    span = max(v_max - v_min, 1)
    label_off = span * 0.04
    ax.set_ylim(v_min - span * 0.08 if v_min < 0 else 0,
                v_max + span * 0.18)

    for bar, v in zip(bars, vals):
        if v >= 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + label_off,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="#334155")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, v - label_off,
                    f"{v:.1f}", ha="center", va="top", fontsize=8, color="#334155")
    ax.axhline(avg, color="#64748b", linestyle="--", linewidth=0.9, alpha=0.7,
               label=f"연 평균 {avg:.1f}%")
    ax.set_ylabel("평균 공실률 (%)", fontsize=10, color="#475569")
    ax.set_title("월별 공실률 패턴", fontsize=12, fontweight="bold", color="#0f172a", loc="left", pad=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#475569")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# BUSINESS PDF — 운영자·경영진용 (reportlab + matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
def generate_business_pdf(output_path: Path = BUSINESS_PDF) -> Path:
    """비즈니스 보고서 PDF 생성 (reportlab).

    톤: 비기술 비즈니스 언어. MAE/RMSE/SHAP/혼동행렬 같은 용어 일절 없음.
    수치적 변화를 매트플롯립 차트로 시각화하여 임베드.
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        KeepTogether,
    )

    font_reg, font_bold = _register_korean_fonts()

    # ── 데이터 로드 ─────────────────────────────────────────
    llm = _load_llm_report()
    long_df = _load_csv(config.LONG_CSV)
    agg = _aggregate_route_vacancy(long_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="KTX 좌석 공실 분석 리포트",
        author="VACANT",
    )

    # ── 스타일 ─────────────────────────────────────────────
    styles = getSampleStyleSheet()
    style_title = ParagraphStyle(
        "title", parent=styles["Title"], fontName=font_bold, fontSize=22,
        textColor=colors.HexColor("#0f172a"), spaceAfter=4, leading=28, alignment=TA_LEFT,
    )
    style_subtitle = ParagraphStyle(
        "subtitle", parent=styles["Normal"], fontName=font_reg, fontSize=9,
        textColor=colors.HexColor("#64748b"), spaceAfter=14, alignment=TA_LEFT,
    )
    style_h2 = ParagraphStyle(
        "h2", parent=styles["Heading2"], fontName=font_bold, fontSize=13,
        textColor=colors.HexColor("#1e3a8a"), spaceBefore=14, spaceAfter=8, leading=18,
    )
    style_body = ParagraphStyle(
        "body", parent=styles["Normal"], fontName=font_reg, fontSize=10,
        textColor=colors.HexColor("#334155"), spaceAfter=6, leading=17, alignment=TA_JUSTIFY,
    )
    style_eyebrow = ParagraphStyle(
        "eyebrow", parent=styles["Normal"], fontName=font_bold, fontSize=8,
        textColor=colors.HexColor("#1d4ed8"), spaceAfter=4,
    )
    style_summary_headline = ParagraphStyle(
        "summary_headline", parent=styles["Normal"], fontName=font_bold, fontSize=14,
        textColor=colors.HexColor("#0f172a"), spaceAfter=8, leading=20,
    )
    style_disclaimer = ParagraphStyle(
        "disclaimer", parent=styles["Normal"], fontName=font_reg, fontSize=8,
        textColor=colors.HexColor("#94a3b8"), spaceAfter=4, leading=12, alignment=TA_JUSTIFY,
    )
    style_card_title = ParagraphStyle(
        "card_title", parent=styles["Normal"], fontName=font_bold, fontSize=10,
        textColor=colors.HexColor("#0f172a"), spaceAfter=3, leading=13,
    )
    style_card_body = ParagraphStyle(
        "card_body", parent=styles["Normal"], fontName=font_reg, fontSize=9,
        textColor=colors.HexColor("#475569"), leading=14, spaceAfter=0,
    )
    style_card_badge = ParagraphStyle(
        "card_badge", parent=styles["Normal"], fontName=font_bold, fontSize=8,
        textColor=colors.HexColor("#1d4ed8"), spaceAfter=2,
    )
    style_action = ParagraphStyle(
        "action", parent=styles["Normal"], fontName=font_reg, fontSize=10,
        textColor=colors.HexColor("#1e293b"), leftIndent=14, bulletIndent=2,
        leading=16, spaceAfter=4,
    )
    style_kpi_label = ParagraphStyle(
        "kpi_label", parent=styles["Normal"], fontName=font_bold, fontSize=8,
        textColor=colors.HexColor("#64748b"), alignment=TA_LEFT, spaceAfter=4,
    )
    style_kpi_value = ParagraphStyle(
        "kpi_value", parent=styles["Normal"], fontName=font_bold, fontSize=20,
        textColor=colors.HexColor("#0f172a"), alignment=TA_LEFT, leading=22,
    )
    style_kpi_value_text = ParagraphStyle(
        "kpi_value_text", parent=styles["Normal"], fontName=font_bold, fontSize=14,
        textColor=colors.HexColor("#0f172a"), alignment=TA_LEFT, leading=18,
    )
    style_kpi_sub = ParagraphStyle(
        "kpi_sub", parent=styles["Normal"], fontName=font_reg, fontSize=8,
        textColor=colors.HexColor("#94a3b8"), alignment=TA_LEFT, spaceBefore=4,
    )
    style_table_note = ParagraphStyle(
        "table_note", parent=styles["Normal"], fontName=font_reg, fontSize=8,
        textColor=colors.HexColor("#94a3b8"), alignment=TA_LEFT, spaceBefore=4,
    )

    today = datetime.now().strftime("%Y년 %m월 %d일")
    story: list = []

    # ── 헤더 ───────────────────────────────────────────────
    header_left = Paragraph(
        f"<font color='#1d4ed8'><b>VACANT</b></font>"
        f"<br/><font size=8 color='#64748b'>KTX 좌석 공실 분석 시스템</font>",
        ParagraphStyle("hl", fontName=font_bold, fontSize=13,
                       textColor=colors.HexColor("#1d4ed8"), leading=16),
    )
    header_right = Paragraph(
        f"<para align=right>"
        f"<font size=9 color='#64748b'>{today} 발행</font>"
        f"<br/><font size=8 color='#94a3b8'><i>운영팀 보고서</i></font></para>",
        ParagraphStyle("hr", fontName=font_reg, fontSize=9, alignment=TA_RIGHT),
    )
    header_tbl = Table([[header_left, header_right]], colWidths=[8 * cm, 9 * cm])
    header_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -1), 2, colors.HexColor("#1d4ed8")),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("KTX 좌석 공실 분석 리포트", style_title))
    story.append(Paragraph(
        "공공데이터 기반 노선별 공실 현황 분석 및 향후 전망",
        style_subtitle,
    ))

    # ── 핵심 요약 배너 ─────────────────────────────────────
    headline = (llm["headline"] if llm and llm.get("headline") else "KTX 노선별 공실률 현황 분석 결과")
    summary = (llm["summary"] if llm and llm.get("summary") else (
        "본 리포트는 KTX 주요 노선의 좌석 공실 현황을 종합 분석한 결과입니다. "
        "노선별 평균 공실률, 계절·시기별 패턴, 향후 수요 전망을 통해 "
        "배차 계획·요금 정책·마케팅 의사결정을 지원합니다."
    ))
    banner = Table([
        [Paragraph("핵심 요약", style_eyebrow)],
        [Paragraph(headline, style_summary_headline)],
        [Paragraph(summary, style_body)],
    ], colWidths=[17 * cm])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#dbeafe")),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (0, 0), 12),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 12),
        ("LINEBEFORE", (0, 0), (0, -1), 4, colors.HexColor("#1d4ed8")),
    ]))
    story.append(banner)
    story.append(Spacer(1, 0.55 * cm))

    # ── KPI 카드 (3개) ─────────────────────────────────────
    if agg is not None and not agg.empty:
        avg_vac = float(agg["mean_vacancy"].mean())
        worst = agg.iloc[0]
        best = agg.iloc[-1]
        n_routes = int(agg[config.ROUTE_COL].nunique())

        def _kpi_card(label, value_block, sub, bg, border):
            inner = Table([
                [Paragraph(label, style_kpi_label)],
                [value_block],
                [Paragraph(sub, style_kpi_sub)],
            ], colWidths=[5.3 * cm])
            inner.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(border)),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (0, 0), 10),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            return inner

        kpi_avg = _kpi_card(
            "전체 평균 공실률",
            Paragraph(f"{avg_vac:.1f}<font size=12 color='#64748b'>%</font>", style_kpi_value),
            f"분석 노선 {n_routes}개 기준",
            "#f8fafc", "#e2e8f0",
        )
        kpi_worst = _kpi_card(
            "가장 높은 공실 노선",
            Paragraph(str(worst[config.ROUTE_COL]), style_kpi_value_text),
            f"평균 {worst['mean_vacancy']:.1f}% 공실",
            "#fff7f7", "#fca5a5",
        )
        kpi_best = _kpi_card(
            "가장 안정적인 노선",
            Paragraph(str(best[config.ROUTE_COL]), style_kpi_value_text),
            f"평균 {best['mean_vacancy']:.1f}% 공실",
            "#f0fdf4", "#86efac",
        )

        kpi_tbl = Table([[kpi_avg, kpi_worst, kpi_best]],
                        colWidths=[5.7 * cm, 5.7 * cm, 5.7 * cm])
        kpi_tbl.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(kpi_tbl)
        story.append(Spacer(1, 0.5 * cm))

    # ── 노선별 공실률 차트 ─────────────────────────────────
    story.append(Paragraph("노선별 공실 현황", style_h2))
    chart_buf = _chart_vacancy_by_route(agg) if agg is not None else None
    if chart_buf is not None:
        story.append(Image(chart_buf, width=17 * cm, height=8.5 * cm))
        story.append(Paragraph(
            "공실률 25% 이상은 주의, 15~25%는 관찰, 15% 미만은 양호로 분류합니다.",
            style_table_note,
        ))
        story.append(Spacer(1, 0.3 * cm))

    # ── 노선별 상세 표 ─────────────────────────────────────
    if agg is not None and not agg.empty:
        header_row = ["노선", "평균 공실률", "최저", "최고", "관측 수", "상태"]
        rows = [header_row]
        for _, r in agg.iterrows():
            vac = float(r["mean_vacancy"])
            status = "주의" if vac >= 25 else ("관찰" if vac >= 15 else "양호")
            rows.append([
                str(r[config.ROUTE_COL]),
                f"{vac:.1f}%",
                f"{r['min_vacancy']:.1f}%",
                f"{r['max_vacancy']:.1f}%",
                f"{int(r['rows'])}",
                status,
            ])
        tbl = Table(rows, colWidths=[3.0 * cm, 3.0 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm])
        ts = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), font_bold),
            ("FONTNAME", (0, 1), (-1, -1), font_reg),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LINEBELOW", (0, 0), (-1, 0), 0, colors.white),
            ("LINEBELOW", (0, 1), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ]
        # 상태별 색상
        for i in range(1, len(rows)):
            status = rows[i][-1]
            if status == "주의":
                ts.append(("TEXTCOLOR", (-1, i), (-1, i), colors.HexColor("#dc2626")))
                ts.append(("FONTNAME", (-1, i), (-1, i), font_bold))
            elif status == "양호":
                ts.append(("TEXTCOLOR", (-1, i), (-1, i), colors.HexColor("#16a34a")))
                ts.append(("FONTNAME", (-1, i), (-1, i), font_bold))
            else:
                ts.append(("TEXTCOLOR", (-1, i), (-1, i), colors.HexColor("#f59e0b")))
                ts.append(("FONTNAME", (-1, i), (-1, i), font_bold))
        tbl.setStyle(TableStyle(ts))
        story.append(tbl)
        story.append(Spacer(1, 0.5 * cm))

    # ── 페이지 분기 ────────────────────────────────────────
    story.append(PageBreak())

    # ── 추세 차트 ──────────────────────────────────────────
    story.append(Paragraph("연도별 공실률 변화 추이", style_h2))
    trend_buf = _chart_vacancy_trend(long_df)
    if trend_buf is not None:
        story.append(Image(trend_buf, width=17 * cm, height=8.5 * cm))
        story.append(Paragraph(
            "공실률이 0% 미만으로 내려간 구간은 좌석 공급보다 수요가 많은 \"초과수요\" 상태를 의미합니다.",
            style_table_note,
        ))
        story.append(Spacer(1, 0.4 * cm))

    # ── 월별 패턴 ──────────────────────────────────────────
    story.append(Paragraph("월별 공실 패턴", style_h2))
    monthly_buf = _chart_monthly_pattern(long_df)
    if monthly_buf is not None:
        story.append(Image(monthly_buf, width=17 * cm, height=6.6 * cm))
        story.append(Paragraph(
            "평균보다 높은 월은 좌석 활용도가 낮아 프로모션 검토가 필요한 시기이며, "
            "낮은 월은 증편·요금 인상 검토 시점입니다.",
            style_table_note,
        ))
        story.append(Spacer(1, 0.4 * cm))

    # ── 인사이트 카드 (LLM 분석) ───────────────────────────
    cards: list[dict] = llm.get("cards", []) if llm else []
    if cards:
        story.append(Paragraph("주요 분석 인사이트", style_h2))
        TAB_LABELS = {
            "tab1": "노선 현황",
            "tab2": "계절·패턴",
            "tab3a": "예측 모델",
            "tab3b": "수요 예측",
            "tab4": "주요 요인",
        }
        SEVERITY = {
            "critical": ("#fff7f7", "#ef4444", "긴급"),
            "warning":  ("#fffbeb", "#f59e0b", "주의"),
            "info":     ("#eff6ff", "#3b82f6", "참고"),
        }
        for c in cards:
            tab_label = TAB_LABELS.get(c.get("tab_id", ""), c.get("tab_id", "분석"))
            bg, accent, badge = SEVERITY.get(c.get("severity", "info"), SEVERITY["info"])
            inner = Table([
                [Paragraph(
                    f"<font color='{accent}'><b>● {badge}</b></font>  "
                    f"<font color='#64748b' size=8>{tab_label}</font>",
                    style_card_badge,
                )],
                [Paragraph(c.get("title", ""), style_card_title)],
                [Paragraph(c.get("body", ""), style_card_body)],
            ], colWidths=[16.6 * cm])
            inner.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(accent)),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (0, 0), 9),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 9),
            ]))
            story.append(KeepTogether([inner, Spacer(1, 0.2 * cm)]))
        story.append(Spacer(1, 0.3 * cm))

    # ── 권고 액션 ──────────────────────────────────────────
    actions: list[str] = llm.get("actions", []) if llm else []
    if actions:
        story.append(Paragraph("운영 권고 사항", style_h2))
        action_rows = []
        for i, action in enumerate(actions, 1):
            num_cell = Paragraph(
                f"<font color='#1d4ed8'><b>{i:02d}</b></font>",
                ParagraphStyle("num", fontName=font_bold, fontSize=14, alignment=TA_CENTER,
                               textColor=colors.HexColor("#1d4ed8")),
            )
            text_cell = Paragraph(action, style_action)
            action_rows.append([num_cell, text_cell])
        atbl = Table(action_rows, colWidths=[1.4 * cm, 15.6 * cm])
        atbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafafa")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#e2e8f0")),
        ]))
        story.append(atbl)
        story.append(Spacer(1, 0.5 * cm))

    # ── 주의사항 ───────────────────────────────────────────
    story.append(Spacer(1, 0.4 * cm))
    disclaimer_box = Table([[Paragraph(
        "본 리포트는 공공데이터 포털 KTX 여객 이용 통계를 기반으로 자동 생성되었습니다. "
        "예측 수치는 통계적 분석 결과이며, 실제 수요는 외부 요인에 따라 달라질 수 있습니다. "
        "의사결정 시 현장 데이터와 병행 검토를 권장합니다.",
        style_disclaimer,
    )], [Paragraph(
        f"<i>생성일: {today}  |  시스템: VACANT v1.0  |  한국철도공사 공실 분석 플랫폼</i>",
        style_disclaimer,
    )]], colWidths=[17 * cm])
    disclaimer_box.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(disclaimer_box)

    doc.build(story)
    log.info("business PDF saved: %s", output_path)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL MARKDOWN — 개발자·데이터사이언티스트용
# ─────────────────────────────────────────────────────────────────────────────
def generate_technical_pdf(output_path: Path | None = None) -> Path:
    """기술 리포트 PDF 생성 — 모델 성능 지표 중심 (개발자·DS용).

    구조:
      1) Performance Overview (TL;DR)
      2) Regression Metrics (CatBoost) — 표 + 차트
      3) Time-Series Metrics (Prophet) — 표 + 차트 + 잔차 분석
      4) Feature Importance (SHAP) — 차트 + 표
      5) Training Data Spec
      6) 성능 분석 및 개선 방향
      7) Reproduction
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image, KeepTogether, PageBreak, Paragraph,
        SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    if output_path is None:
        output_path = config.REPORTS_DIR / "technical_report.pdf"

    font_reg, font_bold = _register_korean_fonts()
    # 모노스페이스 폰트 — 메트릭/코드 표시용 (Courier는 기본 내장)
    font_mono = "Courier"
    font_mono_bold = "Courier-Bold"

    # ── 데이터 로드 ─────────────────────────────────────────
    long_df = _load_csv(config.LONG_CSV)
    shap_per_route = _load_csv(config.PROCESSED_DIR / "shap_per_route.csv")
    model_comparison = _load_csv(config.PROCESSED_DIR / "model_comparison_per_route.csv")
    prophet_metrics_df = _load_csv(config.PROPHET_METRICS_CSV)
    prophet_comparison_df = _load_csv(config.PROCESSED_DIR / "part4_prophet_vacancy_comparison.csv")

    # ── 차트 사전 생성 ─────────────────────────────────────
    chart_regression = _chart_regression_per_route(model_comparison)
    chart_prophet = _chart_prophet_per_route(prophet_metrics_df)
    chart_shap = _chart_shap_importance(shap_per_route)
    chart_residual = _chart_residual_distribution(prophet_comparison_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="VACANT — Model Performance Report",
        author="VACANT",
    )

    # ── 스타일 ─────────────────────────────────────────────
    styles = getSampleStyleSheet()
    S = {
        "title": ParagraphStyle(
            "title", parent=styles["Title"], fontName=font_bold, fontSize=20,
            textColor=colors.HexColor("#0f172a"), spaceAfter=4, leading=24, alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=styles["Normal"], fontName=font_reg, fontSize=9,
            textColor=colors.HexColor("#64748b"), spaceAfter=10, alignment=TA_LEFT,
        ),
        "h2": ParagraphStyle(
            "h2", parent=styles["Heading2"], fontName=font_bold, fontSize=14,
            textColor=colors.HexColor("#1e3a8a"), spaceBefore=14, spaceAfter=8, leading=18,
        ),
        "h3": ParagraphStyle(
            "h3", parent=styles["Heading3"], fontName=font_bold, fontSize=11,
            textColor=colors.HexColor("#0f172a"), spaceBefore=10, spaceAfter=4, leading=15,
        ),
        "body": ParagraphStyle(
            "body", parent=styles["Normal"], fontName=font_reg, fontSize=9.5,
            textColor=colors.HexColor("#334155"), spaceAfter=4, leading=14,
        ),
        "note": ParagraphStyle(
            "note", parent=styles["Normal"], fontName=font_reg, fontSize=8.5,
            textColor=colors.HexColor("#64748b"), spaceAfter=4, leading=12,
        ),
        "mono": ParagraphStyle(
            "mono", parent=styles["Normal"], fontName=font_mono, fontSize=8.5,
            textColor=colors.HexColor("#1e293b"), leading=12,
        ),
        "code_block": ParagraphStyle(
            "code_block", parent=styles["Normal"], fontName=font_mono, fontSize=8.5,
            textColor=colors.HexColor("#0f172a"), backColor=colors.HexColor("#f1f5f9"),
            leftIndent=8, rightIndent=8, leading=13, spaceBefore=4, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=styles["Normal"], fontName=font_reg, fontSize=9.5,
            textColor=colors.HexColor("#334155"), leading=14,
            leftIndent=10, bulletIndent=0, spaceAfter=2,
        ),
        "finding_title": ParagraphStyle(
            "ft", parent=styles["Normal"], fontName=font_bold, fontSize=10,
            textColor=colors.HexColor("#0f172a"), spaceAfter=2, leading=14,
        ),
        "finding_body": ParagraphStyle(
            "fb", parent=styles["Normal"], fontName=font_reg, fontSize=9,
            textColor=colors.HexColor("#334155"), leading=13, spaceAfter=4,
        ),
    }

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story: list = []

    # ── 헤더 ───────────────────────────────────────────────
    story.append(Paragraph("VACANT — Model Performance Report", S["title"]))
    story.append(Paragraph(
        f"KTX 공실률 예측 모델 성능 audit  ·  Generated <font name='{font_mono}'>{now}</font>  "
        f"·  Models: CatBoost (regression), Prophet (time-series)",
        S["subtitle"],
    ))

    # ───────────────────────────────────────────────────────
    # 1. Performance Overview (TL;DR)
    # ───────────────────────────────────────────────────────
    story.append(Paragraph("1. Performance Overview", S["h2"]))
    story.append(_pdf_overview(model_comparison, prophet_metrics_df, shap_per_route,
                                S, font_bold, font_mono, font_reg))

    # ───────────────────────────────────────────────────────
    # 2. Regression Metrics (CatBoost)
    # ───────────────────────────────────────────────────────
    story.append(Paragraph("2. Regression Metrics (CatBoost)", S["h2"]))
    story.append(Paragraph(
        "Target: <b>공실률</b> (%). "
        "평가 방식: time-series CV — expanding window, 노선별 데이터 크기에 따라 n_splits 자동 조정.",
        S["note"],
    ))

    if chart_regression and chart_regression.exists():
        story.append(Spacer(1, 0.2 * cm))
        story.append(Image(str(chart_regression), width=16.5 * cm, height=7.4 * cm))
        story.append(Spacer(1, 0.2 * cm))

    if model_comparison is not None and not model_comparison.empty:
        story.append(Paragraph("2-1. Per-route CV results", S["h3"]))
        story.append(_df_to_pdf_table(model_comparison, font_reg, font_bold, font_mono))

        story.append(Paragraph("2-2. Aggregate stats", S["h3"]))
        story.append(_pdf_regression_aggregate(model_comparison, font_reg, font_bold, font_mono))

        story.append(Paragraph("2-3. Unified vs per-route comparison", S["h3"]))
        for para in _pdf_regression_comparison(model_comparison, S, font_mono):
            story.append(para)

    story.append(Paragraph("2-4. Hyperparameters", S["h3"]))
    story.append(Paragraph(
        f"<font name='{font_mono}'>iterations=500  learning_rate=0.05  depth=6  "
        f"eval_metric=MAE  early_stopping_rounds=50  random_seed=42</font>",
        S["code_block"],
    ))

    story.append(PageBreak())

    # ───────────────────────────────────────────────────────
    # 3. Time-Series Metrics (Prophet)
    # ───────────────────────────────────────────────────────
    story.append(Paragraph("3. Time-Series Metrics (Prophet)", S["h2"]))
    story.append(Paragraph(
        f"Target: 공실률 (%%). horizon={config.ProphetConfig.horizon_months}개월, "
        f"test_split={config.ProphetConfig.test_months}개월 holdout, "
        f"interval_width={config.ProphetConfig.interval_width}. "
        "평가: Walk-forward holdout — 노선별 마지막 12개월을 test로 분리.",
        S["note"],
    ))

    if chart_prophet and chart_prophet.exists():
        story.append(Spacer(1, 0.2 * cm))
        story.append(Image(str(chart_prophet), width=16.5 * cm, height=8.0 * cm))
        story.append(Spacer(1, 0.2 * cm))

    if prophet_metrics_df is not None and not prophet_metrics_df.empty:
        story.append(Paragraph("3-1. Per-route holdout metrics", S["h3"]))
        story.append(_df_to_pdf_table(prophet_metrics_df, font_reg, font_bold, font_mono))

        story.append(Paragraph("3-2. Aggregate stats (pp = percentage point)", S["h3"]))
        story.append(_pdf_prophet_aggregate(prophet_metrics_df, font_reg, font_bold, font_mono))

        for para in _pdf_prophet_bias(prophet_metrics_df, S):
            story.append(para)

    if prophet_comparison_df is not None and not prophet_comparison_df.empty:
        story.append(PageBreak())
        story.append(Paragraph("3-3. Holdout residual analysis", S["h3"]))

        if chart_residual and chart_residual.exists():
            story.append(Image(str(chart_residual), width=16.5 * cm, height=6.6 * cm))
            story.append(Spacer(1, 0.2 * cm))

        for para in _pdf_residual_summary(prophet_comparison_df, S, font_mono):
            story.append(para)

        # Top-5 worst predictions
        if "abs_error" in prophet_comparison_df.columns:
            top5 = prophet_comparison_df.nlargest(5, "abs_error")[
                [c for c in ("yearmonth", "노선", "actual", "predicted", "error", "abs_error")
                 if c in prophet_comparison_df.columns]
            ]
            story.append(Paragraph("Top-5 worst predictions", S["h3"]))
            story.append(_df_to_pdf_table(top5, font_reg, font_bold, font_mono))

    story.append(Paragraph("3-4. Hyperparameters", S["h3"]))
    story.append(Paragraph(
        f"<font name='{font_mono}'>"
        f"horizon_months={config.ProphetConfig.horizon_months}  "
        f"test_months={config.ProphetConfig.test_months}  "
        f"yearly_seasonality={config.ProphetConfig.yearly_seasonality}  "
        f"weekly_seasonality={config.ProphetConfig.weekly_seasonality}  "
        f"interval_width={config.ProphetConfig.interval_width}"
        "</font>",
        S["code_block"],
    ))

    story.append(PageBreak())

    # ───────────────────────────────────────────────────────
    # 4. Feature Importance (SHAP)
    # ───────────────────────────────────────────────────────
    story.append(Paragraph("4. Feature Importance (SHAP)", S["h2"]))
    story.append(Paragraph(
        "CatBoost 회귀 모델의 SHAP value 평균 절대값. 값이 클수록 예측 분산 기여도가 큼.",
        S["note"],
    ))

    if chart_shap and chart_shap.exists():
        story.append(Spacer(1, 0.2 * cm))
        story.append(Image(str(chart_shap), width=16.5 * cm, height=9.0 * cm))
        story.append(Spacer(1, 0.2 * cm))

    if shap_per_route is not None and not shap_per_route.empty \
            and "변수" in shap_per_route.columns and "SHAP중요도" in shap_per_route.columns:
        agg = (
            shap_per_route.groupby("변수", as_index=False)
            .agg(mean_abs_shap=("SHAP중요도", "mean"),
                 std=("SHAP중요도", "std"),
                 max_abs_shap=("SHAP중요도", "max"))
            .sort_values("mean_abs_shap", ascending=False)
        )
        story.append(Paragraph("4-1. Cross-route aggregate", S["h3"]))
        story.append(_df_to_pdf_table(agg, font_reg, font_bold, font_mono))

        story.append(Paragraph("4-2. Top-3 feature stability", S["h3"]))
        for _, r in agg.head(3).iterrows():
            cv = (r["std"] / r["mean_abs_shap"]) if r["mean_abs_shap"] else float("nan")
            txt = (f"• <b>{r['변수']}</b> — "
                   f"mean <font name='{font_mono}'>{r['mean_abs_shap']:.4f}</font>, "
                   f"std <font name='{font_mono}'>{r['std']:.4f}</font>, "
                   f"CV <font name='{font_mono}'>{cv:.1%}</font>")
            story.append(Paragraph(txt, S["bullet"]))
        story.append(Paragraph(
            "CV(coefficient of variation) = std/mean. 작을수록 노선 간 일관된 영향.",
            S["note"],
        ))

    story.append(PageBreak())

    # ───────────────────────────────────────────────────────
    # 5. Training Data Spec
    # ───────────────────────────────────────────────────────
    story.append(Paragraph("5. Training Data Spec", S["h2"]))
    for para in _pdf_training_spec(long_df, S, font_mono):
        story.append(para)

    # ───────────────────────────────────────────────────────
    # 6. 성능 분석 및 개선 방향
    # ───────────────────────────────────────────────────────
    story.append(Paragraph("6. 성능 분석 및 개선 방향", S["h2"]))
    story.append(Paragraph(
        "위 메트릭에서 자동 룰로 감지된 약점과 권장 개선책. 데이터 갱신 시 자동 재계산.",
        S["note"],
    ))

    findings = _collect_findings(
        model_comparison, prophet_metrics_df, prophet_comparison_df,
        shap_per_route, long_df,
    )
    if not findings:
        story.append(Paragraph("✓ 임계값을 넘는 약점이 감지되지 않았습니다.", S["body"]))
    else:
        # 우선순위 표
        sev_order = {"high": 0, "med": 1, "low": 2}
        findings.sort(key=lambda x: sev_order.get(x[0], 99))

        story.append(Paragraph("6-1. Findings overview", S["h3"]))
        story.append(_pdf_findings_overview(findings, font_reg, font_bold))

        story.append(PageBreak())
        story.append(Paragraph("6-2. Detailed actions", S["h3"]))
        for i, (sev, area, finding, action) in enumerate(findings, 1):
            story.append(_pdf_finding_card(i, sev, area, finding, action, S, font_bold, font_reg))
            story.append(Spacer(1, 0.18 * cm))

        story.append(Paragraph("6-3. Priority guide", S["h3"]))
        story.append(Paragraph(
            "<b>High</b> — 다음 sprint 내 처리. 모델 사용성에 직접 영향.<br/>"
            "<b>Med</b> — 다음 model refresh 주기에 포함.<br/>"
            "<b>Low</b> — 기술 부채. 리팩토링 시 정리.",
            S["body"],
        ))

    story.append(PageBreak())

    # ───────────────────────────────────────────────────────
    # 7. Reproduction
    # ───────────────────────────────────────────────────────
    story.append(Paragraph("7. Reproduction", S["h2"]))
    story.append(Paragraph("Re-train + regenerate metrics", S["h3"]))
    story.append(Paragraph(
        f"<font name='{font_mono}'>source .venv/bin/activate<br/>"
        f"python -m src.preprocess_ktx<br/>"
        f"python -m src.features<br/>"
        f"python -m src.Per_route_model<br/>"
        f"python -m src.prophet_model<br/>"
        f"python -m src.shap_analysis<br/>"
        f"python -m src.report_generator</font>",
        S["code_block"],
    ))

    story.append(Paragraph("Report only (no retrain)", S["h3"]))
    story.append(Paragraph(
        f"<font name='{font_mono}'>python -m src.report_generator<br/>"
        f"# or:  curl -X POST http://localhost:8000/api/v1/report/regenerate</font>",
        S["code_block"],
    ))

    story.append(Paragraph("Metric source files", S["h3"]))
    sources = [
        ("data/processed/model_comparison_per_route.csv", "§2 Regression"),
        ("data/processed/part4_prophet_metrics.csv", "§3 Prophet"),
        ("data/processed/part4_prophet_vacancy_comparison.csv", "§3-3 residuals"),
        ("data/processed/shap_per_route.csv", "§4 SHAP"),
        ("data/processed/ktx_long.csv", "§5 Training Data"),
    ]
    src_rows = [["File", "Used in"]] + [[s[0], s[1]] for s in sources]
    src_tbl = Table(src_rows, colWidths=[10.5 * cm, 6.0 * cm])
    src_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), font_bold),
        ("FONTNAME", (0, 1), (0, -1), font_mono),
        ("FONTNAME", (1, 1), (1, -1), font_reg),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
    ]))
    story.append(src_tbl)

    doc.build(story)
    log.info("technical PDF saved: %s", output_path)
    return output_path


# ── 기술 PDF 보조 헬퍼들 ───────────────────────────────────
def _df_to_pdf_table(df: pd.DataFrame, font_reg: str, font_bold: str, font_mono: str,
                     max_col_width_cm: float = 4.0):
    """DataFrame → reportlab Table. 숫자 컬럼은 모노스페이스."""
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    headers = [str(c) for c in df.columns.tolist()]
    rows = [headers]
    for _, r in df.iterrows():
        row = []
        for c in headers:
            v = r[c]
            if isinstance(v, float):
                row.append(f"{v:.4f}")
            elif pd.isna(v):
                row.append("")
            else:
                row.append(str(v))
        rows.append(row)

    # 컬럼 너비 자동 (최대 16.5cm까지)
    n = len(headers)
    total = 16.5
    col_w = [total / n] * n
    # cm to points
    from reportlab.lib.units import cm
    col_widths = [w * cm for w in col_w]

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), font_bold),
        ("FONTNAME", (0, 1), (-1, -1), font_reg),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
    ]
    # 숫자 컬럼은 모노스페이스 적용
    for col_idx, c in enumerate(headers):
        if df[c].dtype.kind in "fi":  # float / int
            style.append(("FONTNAME", (col_idx, 1), (col_idx, -1), font_mono))
            style.append(("ALIGN", (col_idx, 1), (col_idx, -1), "RIGHT"))
    tbl.setStyle(TableStyle(style))
    return tbl


def _pdf_overview(model_comparison, prophet_df, shap_df, S, font_bold, font_mono, font_reg):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib.units import cm

    unified_mae = avg_route_mae = avg_route_rmse = None
    best_route = worst_route = None
    if model_comparison is not None and not model_comparison.empty:
        if "통합모델_MAE" in model_comparison.columns:
            unified_mae = _parse_unified_mae(model_comparison.iloc[0]["통합모델_MAE"])
        if "개별모델_MAE" in model_comparison.columns:
            mae_s = model_comparison["개별모델_MAE"].astype(float)
            avg_route_mae = float(mae_s.mean())
            best_route = model_comparison.loc[mae_s.idxmin(), "노선"]
            worst_route = model_comparison.loc[mae_s.idxmax(), "노선"]
        if "개별모델_RMSE" in model_comparison.columns:
            avg_route_rmse = float(model_comparison["개별모델_RMSE"].astype(float).mean())

    p_mae_avg = p_mae_min = p_mae_max = None
    if prophet_df is not None and not prophet_df.empty and "mae_pp" in prophet_df.columns:
        mae_s = prophet_df["mae_pp"].astype(float)
        p_mae_avg = float(mae_s.mean())
        p_mae_min = float(mae_s.min())
        p_mae_max = float(mae_s.max())

    top_feat = None
    if shap_df is not None and not shap_df.empty \
            and "변수" in shap_df.columns and "SHAP중요도" in shap_df.columns:
        agg = shap_df.groupby("변수")["SHAP중요도"].mean().sort_values(ascending=False)
        if not agg.empty:
            top_feat = (str(agg.index[0]), float(agg.iloc[0]))

    def _fmt(v, fmt="{:.4f}"):
        return fmt.format(v) if v is not None else "n/a"

    rows = [
        ["Model", "Metric", "Value"],
        ["CatBoost (unified)", "MAE (%)", _fmt(unified_mae)],
        ["CatBoost (per-route avg)", "MAE (%)", _fmt(avg_route_mae)],
        ["CatBoost (per-route avg)", "RMSE (%)", _fmt(avg_route_rmse)],
        ["Prophet (per-route avg)", "MAE (pp)", _fmt(p_mae_avg)],
        ["Prophet", "MAE range (pp)",
         f"{_fmt(p_mae_min)} ~ {_fmt(p_mae_max)}"],
    ]
    # 마지막 row(SHAP)는 한글이 들어가므로 mono 적용에서 제외
    last_numeric_row = len(rows) - 1  # SHAP row index (or all numeric rows if no SHAP)
    if top_feat:
        # SHAP row만 일반 폰트로 (한글 포함)
        rows.append(["SHAP top feature", "mean_abs_shap",
                     f"{top_feat[0]}  ({top_feat[1]:.4f})"])

    tbl = Table(rows, colWidths=[6.0 * cm, 4.5 * cm, 6.0 * cm], repeatRows=1)
    tbl_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), font_bold),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (-1, 1), (-1, last_numeric_row), font_mono),
        # SHAP 행은 한글 포함이라 기본 폰트 유지
        ("FONTNAME", (-1, last_numeric_row + 1), (-1, -1), font_reg),
        ("ALIGN", (-1, 1), (-1, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
    ]
    tbl.setStyle(TableStyle(tbl_style))

    # Key observations (한글에는 mono 폰트 X, 숫자만 mono)
    from reportlab.platypus import KeepTogether, Paragraph, Spacer
    obs_paras = []
    if unified_mae is not None and avg_route_mae is not None:
        diff = avg_route_mae - unified_mae
        if diff > 0:
            obs_paras.append(Paragraph(
                f"• 통합 모델(MAE <font name='{font_mono}'>{unified_mae:.4f}</font>)이 "
                f"개별 평균(<font name='{font_mono}'>{avg_route_mae:.4f}</font>)보다 "
                f"<font name='{font_mono}'>{diff:.4f}</font> 우수 → 노선 간 공유 시그널 존재",
                S["bullet"]))
    if best_route and worst_route:
        obs_paras.append(Paragraph(
            f"• 개별 MAE 최고: <b>{best_route}</b> / 최하: <b>{worst_route}</b>",
            S["bullet"]))
    if p_mae_max is not None and avg_route_mae is not None and p_mae_max > avg_route_mae * 2:
        obs_paras.append(Paragraph(
            f"• Prophet 최악 노선 MAE(<font name='{font_mono}'>{p_mae_max:.2f}</font>pp)가 "
            f"회귀 평균의 2배 이상 → 시계열 단독으로는 최근 초과수요 구간 추적 불가",
            S["bullet"]))

    return KeepTogether([tbl, Spacer(1, 0.3 * 28), Paragraph("Key observations", S["h3"]),
                         *obs_paras])


def _pdf_regression_aggregate(model_comparison, font_reg, font_bold, font_mono):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib.units import cm

    mae_s = model_comparison["개별모델_MAE"].astype(float)
    rmse_s = (model_comparison["개별모델_RMSE"].astype(float)
              if "개별모델_RMSE" in model_comparison.columns else None)
    unified = _parse_unified_mae(model_comparison.iloc[0].get("통합모델_MAE"))

    def _rmse(stat_fn):
        return f"{stat_fn(rmse_s):.4f}" if rmse_s is not None else "n/a"

    # 한글 노선명이 들어가는 셀은 Paragraph로 만들어 mono+한글 혼합 가능
    min_route = model_comparison.loc[mae_s.idxmin(), "노선"]
    max_route = model_comparison.loc[mae_s.idxmax(), "노선"]
    mixed_style = ParagraphStyle(
        "mixed_right", fontName=font_reg, fontSize=8.5,
        textColor=colors.HexColor("#1e293b"), alignment=TA_RIGHT,
    )
    min_cell = Paragraph(
        f"<font name='{font_mono}'>{mae_s.min():.4f}</font> ({min_route})", mixed_style)
    max_cell = Paragraph(
        f"<font name='{font_mono}'>{mae_s.max():.4f}</font> ({max_route})", mixed_style)

    unified_cell = f"{unified:.4f}" if unified is not None else "n/a"
    rows = [
        ["Stat", "개별 MAE", "개별 RMSE", "통합 MAE"],
        ["Mean", f"{mae_s.mean():.4f}", _rmse(lambda s: s.mean()), unified_cell],
        ["Median", f"{mae_s.median():.4f}", _rmse(lambda s: s.median()), "—"],
        ["Min", min_cell, _rmse(lambda s: s.min()), "—"],
        ["Max", max_cell, _rmse(lambda s: s.max()), "—"],
        ["Std", f"{mae_s.std():.4f}", _rmse(lambda s: s.std()), "—"],
    ]
    tbl = Table(rows, colWidths=[2.5 * cm, 5.0 * cm, 4.5 * cm, 4.5 * cm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), font_bold),
        ("FONTNAME", (0, 1), (0, -1), font_bold),
        # 1번 컬럼 (개별 MAE)은 Paragraph 셀이 섞여있어 mono 적용 X
        ("FONTNAME", (2, 1), (-1, -1), font_mono),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
    ]))
    return tbl


def _pdf_regression_comparison(model_comparison, S, font_mono):
    from reportlab.platypus import Paragraph

    paras = []
    unified = _parse_unified_mae(model_comparison.iloc[0].get("통합모델_MAE"))
    if unified is None:
        return paras
    mae_s = model_comparison["개별모델_MAE"].astype(float)
    improved = int((mae_s < unified).sum())
    paras.append(Paragraph(
        f"Unified MAE: <font name='{font_mono}'>{unified:.4f}</font>  ·  "
        f"개별 모델 평균: <font name='{font_mono}'>{mae_s.mean():.4f}</font>  "
        f"(Δ <font name='{font_mono}'>{mae_s.mean() - unified:+.4f}</font>)  ·  "
        f"개별 우수 노선: <font name='{font_mono}'>{improved}/{len(mae_s)}</font>",
        S["body"],
    ))
    for _, row in model_comparison.iterrows():
        route_mae = float(row["개별모델_MAE"])
        delta = route_mae - unified
        marker = "<font color='#16a34a'>✓ 개선</font>" if delta < 0 else "<font color='#dc2626'>✗ 후퇴</font>"
        paras.append(Paragraph(
            f"• <b>{row['노선']}</b>: "
            f"per-route <font name='{font_mono}'>{route_mae:.4f}</font> / "
            f"unified <font name='{font_mono}'>{unified:.4f}</font> "
            f"(Δ <font name='{font_mono}'>{delta:+.4f}</font>) → {marker}",
            S["bullet"],
        ))
    return paras


def _pdf_prophet_aggregate(prophet_df, font_reg, font_bold, font_mono):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib.units import cm

    mae_s = prophet_df["mae_pp"].astype(float)
    rmse_s = prophet_df["rmse_pp"].astype(float) if "rmse_pp" in prophet_df.columns else None
    bias_s = prophet_df["mean_error_pp"].astype(float) if "mean_error_pp" in prophet_df.columns else None

    def _cell(s, fn, fmt="{:.4f}"):
        return fmt.format(getattr(s, fn)()) if s is not None else "n/a"

    rows = [["Stat", "MAE pp", "RMSE pp", "Bias (mean_error)"]]
    for name, fn in [("Mean", "mean"), ("Median", "median"), ("Min", "min"),
                     ("Max", "max"), ("Std", "std")]:
        rows.append([name, _cell(mae_s, fn), _cell(rmse_s, fn),
                     _cell(bias_s, fn, "{:+.4f}")])

    tbl = Table(rows, colWidths=[2.5 * cm, 4.5 * cm, 4.5 * cm, 5.0 * cm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), font_bold),
        ("FONTNAME", (0, 1), (0, -1), font_bold),
        ("FONTNAME", (1, 1), (-1, -1), font_mono),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
    ]))
    return tbl


def _pdf_prophet_bias(prophet_df, S):
    from reportlab.platypus import Paragraph
    paras = []
    if "mean_error_pp" not in prophet_df.columns:
        return paras
    bias = float(prophet_df["mean_error_pp"].astype(float).mean())
    direction = "과대예측" if bias > 0 else "과소예측"
    color = "#dc2626" if abs(bias) > 10 else "#475569"
    paras.append(Paragraph(
        f"<b>Bias 해석</b> — 평균 mean_error <font color='{color}'>{bias:+.4f}</font> pp "
        f"→ 전체적으로 <b>{direction}</b> 경향.",
        S["body"],
    ))
    if bias < -10:
        paras.append(Paragraph(
            "최근 초과수요(공실률 &lt; 0) 구간을 따라잡지 못한 결과로 추정.",
            S["note"],
        ))
    return paras


def _pdf_residual_summary(comparison_df, S, font_mono):
    from reportlab.platypus import Paragraph
    paras = []
    if "abs_error" not in comparison_df.columns:
        return paras
    ae = comparison_df["abs_error"].astype(float)
    err = comparison_df["error"].astype(float) if "error" in comparison_df.columns else None
    rmse = (ae ** 2).mean() ** 0.5
    parts = [
        f"전체 잔차 샘플: <font name='{font_mono}'>{len(comparison_df)}</font>",
        f"MAE <font name='{font_mono}'>{ae.mean():.4f}</font>",
        f"RMSE <font name='{font_mono}'>{rmse:.4f}</font>",
        f"잔차 median <font name='{font_mono}'>{ae.median():.4f}</font>",
        f"p75 <font name='{font_mono}'>{ae.quantile(0.75):.4f}</font>",
        f"p95 <font name='{font_mono}'>{ae.quantile(0.95):.4f}</font>",
        f"max <font name='{font_mono}'>{ae.max():.4f}</font>",
    ]
    paras.append(Paragraph(" · ".join(parts), S["body"]))
    if err is not None:
        pos = int((err > 0).sum())
        neg = int((err < 0).sum())
        ratio = neg / max(len(err), 1)
        paras.append(Paragraph(
            f"error 부호: 양(과대예측) <font name='{font_mono}'>{pos}</font> "
            f"/ 음(과소예측) <font name='{font_mono}'>{neg}</font> "
            f"— 과소예측 비율 <font name='{font_mono}'>{ratio:.1%}</font>",
            S["body"],
        ))
    return paras


def _pdf_training_spec(long_df, S, font_mono):
    from reportlab.platypus import Paragraph
    paras = []
    if long_df is None or long_df.empty:
        paras.append(Paragraph("ktx_long.csv 없음", S["body"]))
        return paras
    paras.append(Paragraph(
        f"• Source: <font name='{font_mono}'>data/processed/ktx_long.csv</font>",
        S["bullet"]))
    paras.append(Paragraph(
        f"• Rows: <font name='{font_mono}'>{len(long_df)}</font>",
        S["bullet"]))
    if config.ROUTE_COL in long_df.columns:
        per_route = long_df.groupby(config.ROUTE_COL).size().to_dict()
        # 한글 노선명에 mono 적용하면 깨짐 → 노선명은 bold + 카운트만 mono
        parts = [f"<b>{k}</b>=<font name='{font_mono}'>{v}</font>" for k, v in per_route.items()]
        paras.append(Paragraph(
            f"• Routes: <font name='{font_mono}'>{len(per_route)}</font> — "
            + ", ".join(parts),
            S["bullet"]))
    if "yearmonth" in long_df.columns:
        dt = pd.to_datetime(long_df["yearmonth"], errors="coerce").dropna()
        if not dt.empty:
            paras.append(Paragraph(
                f"• Date range: <font name='{font_mono}'>{dt.min().date()}</font> → "
                f"<font name='{font_mono}'>{dt.max().date()}</font>",
                S["bullet"]))
    if config.TARGET_COL in long_df.columns:
        t = long_df[config.TARGET_COL].astype(float)
        paras.append(Paragraph(
            f"• Target <b>{config.TARGET_COL}</b>: "
            f"mean <font name='{font_mono}'>{t.mean():.2f}%</font>, "
            f"std <font name='{font_mono}'>{t.std():.2f}</font>, "
            f"min <font name='{font_mono}'>{t.min():.2f}%</font>, "
            f"max <font name='{font_mono}'>{t.max():.2f}%</font>, "
            f"negative <font name='{font_mono}'>{(t<0).sum()}/{len(t)} ({(t<0).mean():.1%})</font>",
            S["bullet"]))
    # FEATURE_COLS는 한글 컬럼명이 포함됨 → mono 빼고 bold 처리
    feat_repr = ", ".join(f"<b>{c}</b>" for c in config.FEATURE_COLS)
    paras.append(Paragraph(f"• Feature columns: {feat_repr}", S["bullet"]))
    return paras


def _pdf_findings_overview(findings, font_reg, font_bold):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib.units import cm

    SEV_COLOR = {"high": "#dc2626", "med": "#f59e0b", "low": "#3b82f6"}
    SEV_LABEL = {"high": "● HIGH", "med": "● MED", "low": "● LOW"}

    rows = [["#", "우선순위", "영역", "진단 요약"]]
    for i, (sev, area, finding, _) in enumerate(findings, 1):
        short = finding.replace("\n", " ")
        if len(short) > 90:
            short = short[:90] + "…"
        rows.append([str(i), SEV_LABEL[sev], area, short])

    tbl = Table(rows, colWidths=[0.8 * cm, 2.2 * cm, 2.2 * cm, 11.3 * cm], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), font_bold),
        ("FONTNAME", (0, 1), (-1, -1), font_reg),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
    ]
    for i, (sev, *_) in enumerate(findings, 1):
        style.append(("TEXTCOLOR", (1, i), (1, i), colors.HexColor(SEV_COLOR[sev])))
        style.append(("FONTNAME", (1, i), (1, i), font_bold))
    tbl.setStyle(TableStyle(style))
    return tbl


def _pdf_finding_card(idx, sev, area, finding, action, S, font_bold, font_reg):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib.units import cm

    SEV = {
        "high": ("#fff7f7", "#dc2626", "HIGH"),
        "med":  ("#fffbeb", "#f59e0b", "MED"),
        "low":  ("#eff6ff", "#3b82f6", "LOW"),
    }
    bg, accent, badge = SEV.get(sev, SEV["low"])

    inner = Table([
        [Paragraph(
            f"<font color='{accent}'><b>● {badge}</b></font>  "
            f"<font color='#64748b'>[{idx}] {area}</font>",
            S["finding_title"],
        )],
        [Paragraph(f"<b>Finding</b> — {finding}", S["finding_body"])],
        [Paragraph(f"<b>Action</b> — {action}", S["finding_body"])],
    ], colWidths=[16.5 * cm])
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
        ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(accent)),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (0, 0), 8),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
    ]))
    return inner


def _collect_findings(model_comparison, prophet_df, comparison_df, shap_df, long_df):
    """기존 _md_improvement_directions에서 쓰던 룰 로직을 PDF용으로 추출."""
    findings: list[tuple[str, str, str, str]] = []

    if model_comparison is not None and not model_comparison.empty \
            and "개별모델_MAE" in model_comparison.columns:
        mae_s = model_comparison["개별모델_MAE"].astype(float)
        unified = _parse_unified_mae(model_comparison.iloc[0].get("통합모델_MAE"))
        mae_std = float(mae_s.std())
        mae_mean = float(mae_s.mean())
        if mae_mean and (mae_std / mae_mean) > 0.20:
            worst_route = model_comparison.loc[mae_s.idxmax(), "노선"]
            worst_mae = mae_s.max()
            data_size_info = ""
            if "데이터수(행)" in model_comparison.columns:
                worst_rows = int(model_comparison.loc[mae_s.idxmax(), "데이터수(행)"])
                data_size_info = f" (학습 데이터 {worst_rows}행)"
            findings.append((
                "high", "Regression",
                f"노선별 MAE 표준편차 {mae_std:.2f} (CV {mae_std/mae_mean:.1%})로 불균질. "
                f"{worst_route}{data_size_info}가 평균보다 {(worst_mae - mae_mean):+.2f} 높음.",
                "1) 데이터 적은 노선은 transfer learning (통합모델 fine-tune) 시도. "
                "2) 노선별 하이퍼파라미터 튜닝(Optuna)으로 iterations/depth 차별화. "
                "3) 분포가 다른 노선은 별도 piecewise 모델 검토."
            ))
        if unified is not None and mae_mean - unified > 1.0:
            improved = int((mae_s < unified).sum())
            findings.append((
                "med", "Regression",
                f"통합모델 MAE {unified:.2f} < 개별 평균 {mae_mean:.2f}. "
                f"개별 우수 노선 {improved}/{len(mae_s)}개뿐.",
                "통합모델을 메인 baseline으로 채택, 개별 모델은 MAE 개선되는 1~2개 노선만 hybrid."
            ))
        elif unified is not None and unified - mae_mean > 1.0:
            findings.append((
                "med", "Regression",
                f"개별 평균 MAE {mae_mean:.2f} < 통합 {unified:.2f}. 노선 간 시그널 차이.",
                "개별 모델을 메인으로 채택. 통합모델은 cold-start 보조용."
            ))

    if prophet_df is not None and not prophet_df.empty:
        if "mae_pp" in prophet_df.columns:
            p_mae_avg = float(prophet_df["mae_pp"].astype(float).mean())
            p_mae_max = float(prophet_df["mae_pp"].astype(float).max())
            worst = prophet_df.loc[prophet_df["mae_pp"].astype(float).idxmax(), "노선"]
            if p_mae_avg > 15:
                findings.append((
                    "high", "Prophet",
                    f"Prophet 평균 holdout MAE {p_mae_avg:.2f} pp. 최악 {worst} ({p_mae_max:.2f} pp).",
                    "1) 코로나/SRT 더미를 add_regressor로 주입. "
                    "2) changepoint_prior_scale 상향 (0.05→0.5). "
                    "3) 최근 초과수요 구간 별도 fit ensemble."
                ))
        if "mean_error_pp" in prophet_df.columns:
            bias = float(prophet_df["mean_error_pp"].astype(float).mean())
            if bias < -10:
                findings.append((
                    "high", "Prophet",
                    f"평균 bias {bias:+.2f} pp — 과대예측 경향. 최근 초과수요 구간 추적 실패.",
                    "1) log1p target 변환. 2) lag1/ma3 회귀 모델로 baseline 교체. "
                    "3) n_changepoints 증가 + interval_width 0.95."
                ))
            elif bias > 10:
                findings.append((
                    "high", "Prophet",
                    f"평균 bias {bias:+.2f} pp — 과소예측 경향.",
                    "Prophet trend 평탄. growth='logistic' + cap 검토."
                ))

    if comparison_df is not None and not comparison_df.empty and "error" in comparison_df.columns:
        err = comparison_df["error"].astype(float)
        if len(err) > 0:
            neg_ratio = float((err < 0).mean())
            if neg_ratio > 0.9 or neg_ratio < 0.1:
                direction = "과대예측" if neg_ratio > 0.9 else "과소예측"
                findings.append((
                    "high", "Residual",
                    f"holdout 잔차 부호의 {max(neg_ratio, 1-neg_ratio):.0%}가 한 방향({direction}).",
                    "1) train/test 분포 비교 (KS test). 2) 최근 가중치 ↑ (sample_weight). "
                    "3) 초과수요 분류기 + 회귀 stacked ensemble."
                ))
        if "노선" in comparison_df.columns and "abs_error" in comparison_df.columns:
            grp = comparison_df.groupby("노선")["abs_error"].mean()
            if len(grp) >= 2 and grp.max() > grp.mean() * 1.5:
                worst_route = grp.idxmax()
                findings.append((
                    "med", "Residual",
                    f"{worst_route} 잔차가 평균 대비 {(grp.max()/grp.mean() - 1):.0%} 높음.",
                    f"{worst_route} 단독 분석. 이상치 제거 또는 Huber loss."
                ))

    if shap_df is not None and not shap_df.empty \
            and "변수" in shap_df.columns and "SHAP중요도" in shap_df.columns:
        agg = shap_df.groupby("변수")["SHAP중요도"].agg(["mean", "std"])
        top3 = agg.sort_values("mean", ascending=False).head(3)
        for feat, row in top3.iterrows():
            if row["mean"] and (row["std"] / row["mean"]) > 0.5:
                findings.append((
                    "med", "Features",
                    f"Top feature {feat} CV {row['std']/row['mean']:.0%}.",
                    f"노선 dummy interaction ({feat} × 노선) 또는 노선별 분리 학습 강화."
                ))
        weak = agg[agg["mean"] < 0.1]
        if len(weak) >= 2:
            findings.append((
                "low", "Features",
                f"평균 SHAP < 0.1 feature: {', '.join(weak.index.tolist())}.",
                "feature 제거 또는 재인코딩."
            ))

    if long_df is not None and not long_df.empty and config.ROUTE_COL in long_df.columns:
        per_route_rows = long_df.groupby(config.ROUTE_COL).size()
        if len(per_route_rows) >= 2:
            ratio = per_route_rows.max() / max(per_route_rows.min(), 1)
            if ratio > 2:
                min_route = per_route_rows.idxmin()
                findings.append((
                    "med", "Data",
                    f"노선별 데이터 격차 {ratio:.1f}배 — {min_route} {int(per_route_rows.min())}행.",
                    f"{min_route}: 일별 raw 데이터 활용 또는 oversampling."
                ))

    if long_df is not None and not long_df.empty and config.TARGET_COL in long_df.columns:
        t = long_df[config.TARGET_COL].astype(float)
        neg = (t < 0).sum()
        if len(t) and neg / len(t) > 0.25:
            findings.append((
                "high", "Data",
                f"target 음수(초과수요) 비율 {neg/len(t):.1%}.",
                "2-stage 모델 (분류기 + 회귀) 또는 target normalize."
            ))

    return findings


def generate_technical_markdown(output_path: Path = TECHNICAL_MD) -> Path:
    """기술 리포트 Markdown 생성 — 모델 성능 지표 중심.

    개발자·데이터사이언티스트를 위한 성능 audit 문서.
    데이터 EDA·비즈니스 인사이트는 비즈니스 PDF에 위임.
    """
    long_df = _load_csv(config.LONG_CSV)
    shap_per_route = _load_csv(config.PROCESSED_DIR / "shap_per_route.csv")
    model_comparison = _load_csv(config.PROCESSED_DIR / "model_comparison_per_route.csv")
    prophet_metrics_df = _load_csv(config.PROPHET_METRICS_CSV)
    prophet_comparison_df = _load_csv(config.PROCESSED_DIR / "part4_prophet_vacancy_comparison.csv")

    # 차트 사전 생성 (모델 성능 시각화)
    chart_regression = _chart_regression_per_route(model_comparison)
    chart_prophet = _chart_prophet_per_route(prophet_metrics_df)
    chart_shap = _chart_shap_importance(shap_per_route)
    chart_residual = _chart_residual_distribution(prophet_comparison_df)

    sections = [
        _md_header(),
        _md_toc(),
        _md_performance_overview(model_comparison, prophet_metrics_df, shap_per_route),
        _md_regression_metrics(model_comparison, chart_regression),
        _md_prophet_metrics(prophet_metrics_df, prophet_comparison_df,
                            chart_prophet, chart_residual),
        _md_shap(shap_per_route, chart_shap),
        _md_train_data_spec(long_df),
        _md_improvement_directions(model_comparison, prophet_metrics_df,
                                   prophet_comparison_df, shap_per_route, long_df),
        _md_reproduction(),
    ]
    content = "\n\n".join(filter(None, sections))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    log.info("technical Markdown saved: %s", output_path)
    return output_path


def _md_header() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""# VACANT — Model Performance Report

| | |
|---|---|
| **Generated** | `{now}` |
| **Scope** | KTX 공실률 예측 모델 성능 audit |
| **Models** | CatBoost (regression) · Prophet (time series) |

---"""


def _md_toc() -> str:
    return """## 목차

1. Performance Overview (TL;DR)
2. Regression Metrics (CatBoost)
3. Time-Series Metrics (Prophet)
4. Feature Importance (SHAP)
5. Training Data Spec
6. 성능 분석 및 개선 방향
7. Reproduction

---"""


def _md_performance_overview(
    model_comparison: pd.DataFrame | None,
    prophet_df: pd.DataFrame | None,
    shap_df: pd.DataFrame | None,
) -> str:
    """모든 메트릭의 한눈 요약 (TL;DR)."""
    lines = ["## 1. Performance Overview (TL;DR)\n"]

    # 회귀
    unified_mae = None
    avg_route_mae = None
    avg_route_rmse = None
    best_route = worst_route = None
    if model_comparison is not None and not model_comparison.empty:
        if "통합모델_MAE" in model_comparison.columns:
            unified_mae = _parse_unified_mae(model_comparison.iloc[0]["통합모델_MAE"])
        if "개별모델_MAE" in model_comparison.columns:
            mae_series = model_comparison["개별모델_MAE"].astype(float)
            avg_route_mae = float(mae_series.mean())
            best_route = model_comparison.loc[mae_series.idxmin(), "노선"]
            worst_route = model_comparison.loc[mae_series.idxmax(), "노선"]
        if "개별모델_RMSE" in model_comparison.columns:
            avg_route_rmse = float(model_comparison["개별모델_RMSE"].astype(float).mean())

    # Prophet
    prophet_mae_avg = prophet_mae_min = prophet_mae_max = None
    if prophet_df is not None and not prophet_df.empty and "mae_pp" in prophet_df.columns:
        mae_s = prophet_df["mae_pp"].astype(float)
        prophet_mae_avg = float(mae_s.mean())
        prophet_mae_min = float(mae_s.min())
        prophet_mae_max = float(mae_s.max())

    # SHAP top feature
    top_feat = None
    if shap_df is not None and not shap_df.empty \
            and "변수" in shap_df.columns and "SHAP중요도" in shap_df.columns:
        agg = shap_df.groupby("변수")["SHAP중요도"].mean().sort_values(ascending=False)
        if not agg.empty:
            top_feat = (str(agg.index[0]), float(agg.iloc[0]))

    def _fmt(v, fmt="{:.4f}"):
        return fmt.format(v) if v is not None else "n/a"

    lines.append("| Model | Metric | Value |")
    lines.append("|---|---|---:|")
    lines.append(f"| CatBoost (unified) | MAE (%) | `{_fmt(unified_mae)}` |")
    lines.append(f"| CatBoost (per-route avg) | MAE (%) | `{_fmt(avg_route_mae)}` |")
    lines.append(f"| CatBoost (per-route avg) | RMSE (%) | `{_fmt(avg_route_rmse)}` |")
    lines.append(f"| Prophet (per-route avg) | MAE (pp) | `{_fmt(prophet_mae_avg)}` |")
    lines.append(f"| Prophet | MAE range (pp) | `{_fmt(prophet_mae_min)}` ~ `{_fmt(prophet_mae_max)}` |")
    if top_feat:
        lines.append(f"| SHAP top feature | mean_abs_shap | `{top_feat[0]}` ({_fmt(top_feat[1])}) |")
    lines.append("")

    # 핵심 관찰
    obs = []
    if unified_mae is not None and avg_route_mae is not None:
        diff = avg_route_mae - unified_mae
        if diff > 0:
            obs.append(f"통합 모델(MAE `{unified_mae:.4f}`)이 개별 모델 평균(`{avg_route_mae:.4f}`)보다 "
                       f"`{diff:.4f}` 더 우수 → 노선 간 공유 시그널 존재")
        else:
            obs.append(f"개별 모델 평균(MAE `{avg_route_mae:.4f}`)이 통합 모델(`{unified_mae:.4f}`)보다 "
                       f"`{abs(diff):.4f}` 더 우수 → 노선별 분리 학습 권고")
    if best_route and worst_route:
        obs.append(f"개별 MAE 최고: `{best_route}` / 최하: `{worst_route}`")
    if prophet_mae_max is not None and avg_route_mae is not None and prophet_mae_max > avg_route_mae * 2:
        obs.append(f"Prophet 최악 노선 MAE(`{prophet_mae_max:.2f}`pp)가 회귀 평균(`{avg_route_mae:.2f}`)의 "
                   "2배 이상 → 시계열 단독으로는 최근 초과수요 구간을 추적 불가")

    if obs:
        lines.append("### Key observations\n")
        for o in obs:
            lines.append(f"- {o}")

    return "\n".join(lines)


def _md_regression_metrics(model_comparison: pd.DataFrame | None,
                           chart_path: Path | None = None) -> str:
    """회귀 모델 (CatBoost) 상세 메트릭."""
    lines = ["## 2. Regression Metrics (CatBoost)\n"]
    lines.append("> Target: `공실률` (%). 학습 데이터: `data/processed/ktx_model_input.csv` (노선별 long format).\n"
                 "> 평가 방식: time-series CV — expanding window, 노선별 데이터 크기에 따라 `n_splits` 자동 조정.")
    lines.append("")

    if model_comparison is None or model_comparison.empty:
        lines.append("> 미생성 — `python -m src.Per_route_model`")
        return "\n".join(lines)

    chart_md = _md_chart_link(chart_path, "Per-route CatBoost MAE/RMSE with unified baseline")
    if chart_md:
        lines.append(chart_md)
        lines.append("")

    # 본 표
    lines.append("### 2-1. Per-route CV results\n")
    lines.append(model_comparison.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    # 통계 요약
    if "개별모델_MAE" in model_comparison.columns:
        mae_s = model_comparison["개별모델_MAE"].astype(float)
        rmse_s = model_comparison["개별모델_RMSE"].astype(float) if "개별모델_RMSE" in model_comparison.columns else None
        unified_mae = _parse_unified_mae(model_comparison.iloc[0].get("통합모델_MAE"))

        lines.append("### 2-2. Aggregate stats\n")
        lines.append("| Stat | 개별모델 MAE | 개별모델 RMSE | 통합모델 MAE |")
        lines.append("|---|---:|---:|---:|")

        def _rmse_cell(stat_fn):
            return f"`{stat_fn(rmse_s):.4f}`" if rmse_s is not None else "n/a"

        unified_cell = f"`{unified_mae:.4f}`" if unified_mae is not None else "n/a"
        lines.append(f"| Mean | `{mae_s.mean():.4f}` | {_rmse_cell(lambda s: s.mean())} | {unified_cell} |")
        lines.append(f"| Median | `{mae_s.median():.4f}` | {_rmse_cell(lambda s: s.median())} | — |")
        lines.append(f"| Min | `{mae_s.min():.4f}` ({model_comparison.loc[mae_s.idxmin(), '노선']}) "
                     f"| {_rmse_cell(lambda s: s.min())} | — |")
        lines.append(f"| Max | `{mae_s.max():.4f}` ({model_comparison.loc[mae_s.idxmax(), '노선']}) "
                     f"| {_rmse_cell(lambda s: s.max())} | — |")
        lines.append(f"| Std | `{mae_s.std():.4f}` | {_rmse_cell(lambda s: s.std())} | — |")
        lines.append("")

        # 비교 해석
        lines.append("### 2-3. Unified vs per-route comparison\n")
        if unified_mae is not None:
            lines.append(f"- **Unified MAE**: `{unified_mae:.4f}`")
            lines.append(f"- **Per-route MAE mean**: `{mae_s.mean():.4f}` (Δ `{mae_s.mean() - unified_mae:+.4f}`)")
            improved = (mae_s < unified_mae).sum()
            lines.append(f"- 개별 모델이 통합 모델보다 우수한 노선: `{improved}/{len(mae_s)}`개")
            for _, row in model_comparison.iterrows():
                route_mae = float(row["개별모델_MAE"])
                delta = route_mae - unified_mae
                marker = "✓ 개선" if delta < 0 else "✗ 후퇴"
                lines.append(f"  - `{row['노선']}`: per-route MAE `{route_mae:.4f}` / unified `{unified_mae:.4f}` "
                             f"(Δ `{delta:+.4f}`) → {marker}")

    # 하이퍼파라미터
    lines.append("")
    lines.append("### 2-4. Hyperparameters (`src/config.py::CatBoostConfig`)\n")
    lines.append("```python")
    lines.append(f"iterations            = {config.CatBoostConfig.iterations}")
    lines.append(f"learning_rate         = {config.CatBoostConfig.learning_rate}")
    lines.append(f"depth                 = {config.CatBoostConfig.depth}")
    lines.append(f"eval_metric           = {config.CatBoostConfig.eval_metric}")
    lines.append(f"early_stopping_rounds = {config.CatBoostConfig.early_stopping_rounds}")
    lines.append(f"random_seed           = {config.CatBoostConfig.random_seed}")
    lines.append("```")

    return "\n".join(lines)


def _md_prophet_metrics(
    prophet_df: pd.DataFrame | None,
    comparison_df: pd.DataFrame | None,
    chart_metrics: Path | None = None,
    chart_residual: Path | None = None,
) -> str:
    """Prophet 시계열 모델 메트릭 + 잔차 분석."""
    lines = ["## 3. Time-Series Metrics (Prophet)\n"]
    lines.append(f"> Target: `공실률` (%). horizon=`{config.ProphetConfig.horizon_months}`개월, "
                 f"test_split=`{config.ProphetConfig.test_months}`개월 holdout, "
                 f"interval_width=`{config.ProphetConfig.interval_width}`.\n"
                 "> 평가 방식: Walk-forward holdout — 노선별 마지막 12개월을 test로 분리.")
    lines.append("")

    if prophet_df is None or prophet_df.empty:
        lines.append("> 미생성 — `python -m src.prophet_model`")
        return "\n".join(lines)

    chart_md = _md_chart_link(chart_metrics, "Prophet per-route MAE/RMSE/Bias")
    if chart_md:
        lines.append(chart_md)
        lines.append("")

    # 메트릭 표
    lines.append("### 3-1. Per-route holdout metrics\n")
    lines.append(prophet_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    if "mae_pp" in prophet_df.columns:
        mae_s = prophet_df["mae_pp"].astype(float)
        rmse_s = prophet_df["rmse_pp"].astype(float) if "rmse_pp" in prophet_df.columns else None
        bias_s = prophet_df["mean_error_pp"].astype(float) if "mean_error_pp" in prophet_df.columns else None

        lines.append("### 3-2. Aggregate stats (pp = percentage point)\n")
        lines.append("| Stat | MAE pp | RMSE pp | Bias (mean_error) |")
        lines.append("|---|---:|---:|---:|")
        for stat_name, fn in [("Mean", "mean"), ("Median", "median"),
                              ("Min", "min"), ("Max", "max"), ("Std", "std")]:
            mae_v = getattr(mae_s, fn)()
            rmse_v = getattr(rmse_s, fn)() if rmse_s is not None else None
            bias_v = getattr(bias_s, fn)() if bias_s is not None else None
            rmse_cell = f"`{rmse_v:.4f}`" if rmse_v is not None else "n/a"
            bias_cell = f"`{bias_v:+.4f}`" if bias_v is not None else "n/a"
            lines.append(f"| {stat_name} | `{mae_v:.4f}` | {rmse_cell} | {bias_cell} |")
        lines.append("")

        # bias 해석
        if bias_s is not None:
            avg_bias = float(bias_s.mean())
            sign = "과대예측" if avg_bias > 0 else "과소예측"
            lines.append(f"**Bias 해석**: 평균 `mean_error = {avg_bias:+.4f}` pp → 전체적으로 `{sign}` 경향.")
            if avg_bias < -10:
                lines.append("→ 모델이 실제보다 공실률을 크게 낮게 예측. 최근 초과수요(공실률 < 0) 구간을 따라잡지 못한 결과로 추정.")
            lines.append("")

    # 잔차 분석
    if comparison_df is not None and not comparison_df.empty \
            and "abs_error" in comparison_df.columns:
        lines.append("### 3-3. Holdout residual analysis (`part4_prophet_vacancy_comparison.csv`)\n")
        residual_md = _md_chart_link(chart_residual, "Residual distribution + per-route boxplot")
        if residual_md:
            lines.append(residual_md)
            lines.append("")
        ae = comparison_df["abs_error"].astype(float)
        err = comparison_df["error"].astype(float) if "error" in comparison_df.columns else None
        lines.append(f"- 전체 잔차 샘플: `{len(comparison_df)}`")
        lines.append(f"- MAE: `{ae.mean():.4f}` / RMSE: `{(ae**2).mean()**0.5:.4f}`")
        lines.append(f"- 잔차 분포: median `{ae.median():.4f}`, "
                     f"p75 `{ae.quantile(0.75):.4f}`, p95 `{ae.quantile(0.95):.4f}`, max `{ae.max():.4f}`")
        if err is not None:
            pos = (err > 0).sum()
            neg = (err < 0).sum()
            lines.append(f"- error 부호 분포: 양(`+`, 과대예측) `{pos}` / 음(`-`, 과소예측) `{neg}` "
                         f"(과소예측 비율 `{neg / max(len(err),1):.1%}`)")
        lines.append("")
        # 노선별 최악 5개
        if "노선" in comparison_df.columns:
            top_err = comparison_df.nlargest(5, "abs_error")[
                [c for c in ("yearmonth", "노선", "actual", "predicted", "error", "abs_error") if c in comparison_df.columns]
            ]
            lines.append("**Top-5 worst predictions**\n")
            lines.append(top_err.to_markdown(index=False, floatfmt=".4f"))

    # 하이퍼파라미터
    lines.append("")
    lines.append("### 3-4. Hyperparameters (`src/config.py::ProphetConfig`)\n")
    lines.append("```python")
    lines.append(f"horizon_months      = {config.ProphetConfig.horizon_months}")
    lines.append(f"test_months         = {config.ProphetConfig.test_months}")
    lines.append(f"yearly_seasonality  = {config.ProphetConfig.yearly_seasonality}")
    lines.append(f"weekly_seasonality  = {config.ProphetConfig.weekly_seasonality}")
    lines.append(f"daily_seasonality   = {config.ProphetConfig.daily_seasonality}")
    lines.append(f"interval_width      = {config.ProphetConfig.interval_width}")
    lines.append("```")

    return "\n".join(lines)


def _md_train_data_spec(long_df: pd.DataFrame | None) -> str:
    """학습 데이터 스펙 (행 수, 기간, 노선) — 메트릭 재현에 필요한 최소 정보."""
    lines = ["## 5. Training Data Spec\n"]
    if long_df is None or long_df.empty:
        lines.append("> `ktx_long.csv` 없음")
        return "\n".join(lines)

    lines.append(f"- Source: `data/processed/ktx_long.csv`")
    lines.append(f"- Rows: `{len(long_df)}`")
    if config.ROUTE_COL in long_df.columns:
        per_route = long_df.groupby(config.ROUTE_COL).size().to_dict()
        lines.append(f"- Routes: `{len(per_route)}` — {per_route}")
    if "yearmonth" in long_df.columns:
        dt = pd.to_datetime(long_df["yearmonth"], errors="coerce").dropna()
        if not dt.empty:
            lines.append(f"- Date range: `{dt.min().date()}` → `{dt.max().date()}` "
                         f"(`{(dt.max() - dt.min()).days}` days)")
    if config.TARGET_COL in long_df.columns:
        t = long_df[config.TARGET_COL].astype(float)
        lines.append(f"- Target `{config.TARGET_COL}`: "
                     f"mean `{t.mean():.2f}%`, std `{t.std():.2f}`, "
                     f"min `{t.min():.2f}%`, max `{t.max():.2f}%`, "
                     f"negative(`<0`) `{(t<0).sum()}`/`{len(t)}` "
                     f"(`{(t<0).mean():.1%}`)")
    lines.append("")
    lines.append(f"- Feature columns: `{list(config.FEATURE_COLS)}`")
    lines.append(f"- Categorical: `{list(config.CATEGORICAL_FEATURES)}`")
    lines.append(f"- Lag features: `{list(config.LAG_FEATURES)}`")
    return "\n".join(lines)


def _md_shap(shap_df: pd.DataFrame | None,
             chart_path: Path | None = None) -> str:
    """SHAP — 모델 해석성 지표."""
    lines = ["## 4. Feature Importance (SHAP)\n"]
    lines.append("> CatBoost 회귀 모델의 SHAP value 평균 절대값. 값이 클수록 예측 분산 기여도가 큼.\n"
                 "> Source: `data/processed/shap_per_route.csv`")
    lines.append("")

    if shap_df is None or shap_df.empty:
        lines.append("> 미생성 — `python -m src.shap_analysis`")
        return "\n".join(lines)

    chart_md = _md_chart_link(chart_path, "SHAP feature importance (mean + std across routes)")
    if chart_md:
        lines.append(chart_md)
        lines.append("")

    if "변수" in shap_df.columns and "SHAP중요도" in shap_df.columns:
        agg = (
            shap_df.groupby("변수", as_index=False)
            .agg(mean_abs_shap=("SHAP중요도", "mean"),
                 std=("SHAP중요도", "std"),
                 min=("SHAP중요도", "min"),
                 max=("SHAP중요도", "max"))
            .sort_values("mean_abs_shap", ascending=False)
        )
        lines.append("### 4-1. Cross-route aggregate\n")
        lines.append(agg.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

        # Top-N feature stability check
        top3 = agg.head(3)
        lines.append("### 4-2. Top-3 feature stability\n")
        for _, r in top3.iterrows():
            cv = (r["std"] / r["mean_abs_shap"]) if r["mean_abs_shap"] else float("nan")
            lines.append(f"- `{r['변수']}` — mean `{r['mean_abs_shap']:.4f}`, "
                         f"std `{r['std']:.4f}`, CV `{cv:.2%}` "
                         f"(min `{r['min']:.4f}` ~ max `{r['max']:.4f}`)")
        lines.append("")
        lines.append("> CV(coefficient of variation) = std/mean. 작을수록 노선 간 일관된 영향.")
        lines.append("")

        # Per-route ranking pivot
        lines.append("### 4-3. Per-route SHAP matrix\n")
        if "노선" in shap_df.columns:
            pivot = shap_df.pivot_table(
                index="변수", columns="노선", values="SHAP중요도", aggfunc="first"
            ).sort_index()
            lines.append(pivot.to_markdown(floatfmt=".4f"))

    return "\n".join(lines)




def _md_reproduction() -> str:
    return """## 7. Reproduction

### Re-train + regenerate metrics

```bash
source .venv/bin/activate
python -m src.preprocess_ktx        # raw xlsx → wide/long CSV
python -m src.features              # feature engineering (lag, MA, season)
python -m src.Per_route_model       # CatBoost per-route + unified, dumps model_comparison_per_route.csv
python -m src.prophet_model         # Prophet holdout, dumps part4_prophet_metrics.csv
python -m src.shap_analysis         # SHAP per route, dumps shap_per_route.csv
python -m src.report_generator      # this file + PDF
```

### Report only (no retrain)

```bash
python -m src.report_generator
# or via API
curl -X POST http://localhost:8000/api/v1/report/regenerate
```

### Metric source files

| File | Used in |
|---|---|
| `data/processed/model_comparison_per_route.csv` | §2 Regression |
| `data/processed/part4_prophet_metrics.csv` | §3 Prophet |
| `data/processed/part4_prophet_vacancy_comparison.csv` | §3-3 residuals |
| `data/processed/shap_per_route.csv` | §4 SHAP |
| `data/processed/ktx_long.csv` | §5 Training Data |

### Runtime

| 항목 | 버전 |
|---|---|
| Python | 3.10+ |
| 의존성 | `requirements.txt` |
| 한국어 폰트 | Pretendard (auto OTF→TTF via cu2qu) |
| PDF 엔진 | reportlab + matplotlib |"""


def _md_improvement_directions(
    model_comparison: pd.DataFrame | None,
    prophet_df: pd.DataFrame | None,
    comparison_df: pd.DataFrame | None,
    shap_df: pd.DataFrame | None,
    long_df: pd.DataFrame | None,
) -> str:
    """현재 메트릭을 룰 기반으로 자동 진단 + 구체적 개선 방향 제시.

    각 모델·피처·잔차 패턴에서 임계값을 넘기면 해당 finding과 action을 출력.
    """
    findings: list[tuple[str, str, str, str]] = []
    # (severity, area, finding, recommended_action)

    # ── (1) CatBoost: unified vs per-route ─────────────────
    if model_comparison is not None and not model_comparison.empty:
        if "개별모델_MAE" in model_comparison.columns:
            mae_s = model_comparison["개별모델_MAE"].astype(float)
            unified = _parse_unified_mae(model_comparison.iloc[0].get("통합모델_MAE"))

            # per-route MAE 편차가 크면
            mae_std = float(mae_s.std())
            mae_mean = float(mae_s.mean())
            if mae_mean and (mae_std / mae_mean) > 0.20:
                worst_route = model_comparison.loc[mae_s.idxmax(), "노선"]
                worst_mae = mae_s.max()
                rows_col = "데이터수(행)"
                data_size_info = ""
                if rows_col in model_comparison.columns:
                    worst_rows = int(model_comparison.loc[mae_s.idxmax(), rows_col])
                    data_size_info = f" (학습 데이터 `{worst_rows}`행)"
                findings.append((
                    "high", "Regression",
                    f"노선별 MAE 표준편차 `{mae_std:.2f}` (CV `{mae_std/mae_mean:.1%}`)로 불균질. "
                    f"`{worst_route}`{data_size_info} 가 평균보다 `{(worst_mae - mae_mean):+.2f}` 높음.",
                    "1) 데이터 적은 노선은 transfer learning (통합모델 fine-tune) 시도. "
                    "2) 노선별 하이퍼파라미터 튜닝(Optuna)으로 `iterations`/`depth` 차별화. "
                    "3) 동해선·경전선처럼 분포가 다른 노선은 별도 piecewise 모델 검토."
                ))

            # 통합 모델이 평균보다 훨씬 좋다면
            if unified is not None and mae_mean - unified > 1.0:
                improved = int((mae_s < unified).sum())
                findings.append((
                    "med", "Regression",
                    f"통합모델 MAE `{unified:.2f}` < 개별모델 평균 `{mae_mean:.2f}`. "
                    f"개별 모델이 더 우수한 노선은 `{improved}/{len(mae_s)}`개뿐.",
                    "통합모델을 메인 baseline으로 채택하고, 개별 모델은 "
                    "MAE가 개선되는 1~2개 노선에만 적용하는 hybrid 구조 권고."
                ))
            elif unified is not None and unified - mae_mean > 1.0:
                findings.append((
                    "med", "Regression",
                    f"개별모델 평균 MAE `{mae_mean:.2f}` < 통합모델 `{unified:.2f}`. "
                    "노선 간 시그널이 달라 통합 학습이 오히려 손해.",
                    "개별 모델을 메인으로 채택. 통합모델은 cold-start 보조용으로만 유지."
                ))

    # ── (2) Prophet: bias / MAE / 회귀 대비 격차 ───────────
    if prophet_df is not None and not prophet_df.empty:
        if "mae_pp" in prophet_df.columns:
            p_mae_avg = float(prophet_df["mae_pp"].astype(float).mean())
            p_mae_max = float(prophet_df["mae_pp"].astype(float).max())
            worst = prophet_df.loc[prophet_df["mae_pp"].astype(float).idxmax(), "노선"]
            if p_mae_avg > 15:
                findings.append((
                    "high", "Prophet",
                    f"Prophet 평균 holdout MAE `{p_mae_avg:.2f}` pp — 시계열 단일 모델로는 부정확. "
                    f"최악 노선 `{worst}` (`{p_mae_max:.2f}` pp).",
                    "1) 코로나/SRT 더미를 Prophet `add_regressor`로 명시 주입. "
                    "2) `changepoint_prior_scale` 상향 (0.05→0.5)으로 구조 변화 민감도 ↑. "
                    "3) 최근 초과수요 구간(2023~)을 별도 fit하여 ensemble."
                ))

        if "mean_error_pp" in prophet_df.columns:
            bias = float(prophet_df["mean_error_pp"].astype(float).mean())
            if bias < -10:
                findings.append((
                    "high", "Prophet",
                    f"평균 bias `{bias:+.2f}` pp — 전체적으로 과대예측 (실제 공실률보다 모델이 높게 예측). "
                    "최근 초과수요(공실률 < 0) 구간을 추적 못하는 패턴.",
                    "1) 학습 target에 `np.log1p` 또는 분포 정규화 시도. "
                    "2) Prophet 대신 `공실률_lag1/ma3` 활용한 회귀 모델을 시계열 baseline으로 교체. "
                    "3) `n_changepoints` 증가 + `interval_width` 0.95로 안전 마진 확보."
                ))
            elif bias > 10:
                findings.append((
                    "high", "Prophet",
                    f"평균 bias `{bias:+.2f}` pp — 전체적으로 과소예측.",
                    "Prophet trend가 너무 평탄. `growth='logistic'` + cap 설정 검토."
                ))

    # ── (3) 잔차 패턴: 부호 편향 ──────────────────────────
    if comparison_df is not None and not comparison_df.empty and "error" in comparison_df.columns:
        err = comparison_df["error"].astype(float)
        if len(err) > 0:
            neg_ratio = float((err < 0).mean())
            if neg_ratio > 0.9 or neg_ratio < 0.1:
                direction = "과대예측" if neg_ratio > 0.9 else "과소예측"
                findings.append((
                    "high", "Residual",
                    f"holdout 잔차 부호의 `{max(neg_ratio, 1-neg_ratio):.0%}`가 한 방향(`{direction}`) — "
                    "분포 미스매치 신호.",
                    "1) train/test 분포 비교 (KS test) — 분포가 다르면 holdout window 재설정. "
                    "2) 최근 데이터 가중치 증가 (sample_weight) 또는 weighted loss. "
                    "3) `초과수요` 분류기를 추가해 회귀와 stacked ensemble 구성."
                ))

        # 노선별 잔차 집중도
        if "노선" in comparison_df.columns and "abs_error" in comparison_df.columns:
            grp = comparison_df.groupby("노선")["abs_error"].mean()
            if len(grp) >= 2 and grp.max() > grp.mean() * 1.5:
                worst_route = grp.idxmax()
                findings.append((
                    "med", "Residual",
                    f"`{worst_route}` 잔차가 평균 대비 `{(grp.max()/grp.mean() - 1):.0%}` 높음 — "
                    "오차가 특정 노선에 집중.",
                    f"`{worst_route}` 단독 분석: 학습 기간 외 이벤트(노선 개통/폐선) 확인, "
                    "이상치 제거 또는 robust loss(`Huber`) 적용."
                ))

    # ── (4) SHAP: 피처 불안정 ──────────────────────────────
    if shap_df is not None and not shap_df.empty \
            and "변수" in shap_df.columns and "SHAP중요도" in shap_df.columns:
        agg = shap_df.groupby("변수")["SHAP중요도"].agg(["mean", "std"])
        top3 = agg.sort_values("mean", ascending=False).head(3)
        for feat, row in top3.iterrows():
            if row["mean"] and (row["std"] / row["mean"]) > 0.5:
                findings.append((
                    "med", "Features",
                    f"Top feature `{feat}` CV `{row['std']/row['mean']:.0%}` (mean `{row['mean']:.2f}`, "
                    f"std `{row['std']:.2f}`) — 노선마다 영향 크게 다름.",
                    f"노선 dummy interaction (e.g., `{feat} × 노선`) 추가 또는 노선별 분리 학습 강화."
                ))

        # 사용 안 되는 약한 feature
        weak = agg[agg["mean"] < 0.1]
        if len(weak) >= 2:
            weak_list = ", ".join(f"`{f}`" for f in weak.index.tolist())
            findings.append((
                "low", "Features",
                f"평균 SHAP < 0.1인 feature: {weak_list} — 학습에 거의 기여 안 함.",
                "feature 제거하여 학습 속도 ↑, 또는 다른 형태(연속/카테고리 인코딩)로 재구성."
            ))

    # ── (5) 데이터 불균형 ──────────────────────────────────
    if long_df is not None and not long_df.empty and config.ROUTE_COL in long_df.columns:
        per_route_rows = long_df.groupby(config.ROUTE_COL).size()
        if len(per_route_rows) >= 2:
            ratio = per_route_rows.max() / max(per_route_rows.min(), 1)
            if ratio > 2:
                min_route = per_route_rows.idxmin()
                findings.append((
                    "med", "Data",
                    f"노선별 데이터 양 격차 `{ratio:.1f}배` — `{min_route}` `{int(per_route_rows.min())}`행 vs "
                    f"max `{int(per_route_rows.max())}`행.",
                    f"`{min_route}` 등 데이터 적은 노선: "
                    "1) 일별 raw 데이터를 활용해 월 단위 외에 추가 frequency 학습. "
                    "2) SMOTE 등 시계열용 augmentation 또는 분포 매칭 oversampling."
                ))

    # ── (6) Target 분포 불균형 ─────────────────────────────
    if long_df is not None and not long_df.empty and config.TARGET_COL in long_df.columns:
        t = long_df[config.TARGET_COL].astype(float)
        neg = (t < 0).sum()
        if len(t) and neg / len(t) > 0.25:
            findings.append((
                "high", "Data",
                f"target 음수(`초과수요`) 비율 `{neg/len(t):.1%}` — 클래스/구간 불균형.",
                "회귀 → '음수/양수 분류기 + 회귀' 2-stage 모델 검토. "
                "또는 target을 `tanh` 등 양방향 normalize."
            ))

    # ── 출력 ───────────────────────────────────────────────
    lines = ["## 6. 성능 분석 및 개선 방향\n"]
    lines.append("> 위 메트릭에서 자동 룰로 감지된 약점과 권장 개선책. "
                 "데이터가 갱신되면 자동 재계산됩니다.\n")

    if not findings:
        lines.append("✅ 임계값을 넘는 약점이 감지되지 않았습니다. 현재 모델은 안정적.")
        return "\n".join(lines)

    # severity 순서로 정렬
    sev_order = {"high": 0, "med": 1, "low": 2}
    sev_badge = {"high": "🔴 High", "med": "🟡 Med", "low": "🔵 Low"}
    findings.sort(key=lambda x: sev_order.get(x[0], 99))

    # 요약 표
    lines.append("### 6-1. Findings overview\n")
    lines.append("| # | 우선순위 | 영역 | 진단 요약 |")
    lines.append("|---:|---|---|---|")
    for i, (sev, area, finding, _) in enumerate(findings, 1):
        # 표 셀에서 깨지지 않게: 줄바꿈/파이프 제거 + 100자 컷
        flat = finding.replace("\n", " ").replace("|", "/")
        short = flat[:100] + ("…" if len(flat) > 100 else "")
        lines.append(f"| {i} | {sev_badge[sev]} | {area} | {short} |")
    lines.append("")

    # 상세
    lines.append("### 6-2. Detailed actions\n")
    for i, (sev, area, finding, action) in enumerate(findings, 1):
        lines.append(f"#### [{i}] {sev_badge[sev]} · {area}\n")
        lines.append(f"**Finding** — {finding}\n")
        lines.append(f"**Recommended action** — {action}\n")

    # 우선순위 가이드
    lines.append("### 6-3. Priority guide\n")
    lines.append("- 🔴 **High**: 다음 sprint 내 처리 권장. 모델 사용성에 직접 영향.")
    lines.append("- 🟡 **Med**: 다음 model refresh 주기(월 단위)에 포함.")
    lines.append("- 🔵 **Low**: 기술 부채. 리팩토링 시 정리.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
def main() -> None:
    log.info("generating business PDF...")
    biz_pdf = generate_business_pdf()

    log.info("generating technical PDF...")
    tech_pdf = generate_technical_pdf()

    log.info("=== Reports generated ===")
    log.info("Business  (PDF): %s", biz_pdf)
    log.info("Technical (PDF): %s", tech_pdf)


if __name__ == "__main__":
    main()
