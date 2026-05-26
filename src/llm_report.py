"""Part 6: LLM-powered natural language insight report generation.

Consumes structured analysis results from Parts 1~5 and produces:
  • Per-tab insight cards (1~2 sentences, displayed at top of each dashboard tab)
  • Daily executive summary report (5~10 sentences + 3 recommended actions)
  • Anomaly alerts (when metrics deviate beyond threshold)

Provider-agnostic design: switch between Anthropic Claude, OpenAI GPT, or local
LLM by changing LLM_PROVIDER env var. Defaults to Anthropic Claude.

Resilient by design: if API call fails, falls back to template-based summaries
so the dashboard always has *some* text to display.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from . import config
from .logging_setup import get_logger


log = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# Structured input/output schemas
# ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AnalysisSnapshot:
    """Bundle of analysis results passed to the LLM.

    Each field corresponds to a dashboard tab (Parts 1~5). Optional fields
    can be None when a tab has no fresh data.
    """

    generated_at: str
    part1_route_aggregate: list[dict] = field(default_factory=list)
    part2_pattern_summary: dict | None = None
    part3a_regression_metrics: dict | None = None
    part3b_prophet_forecast: list[dict] = field(default_factory=list)
    part5_shap_importance: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class InsightCard:
    """Single-tab insight rendered at the top of each dashboard tab."""

    tab_id: str  # 'tab1' | 'tab2' | 'tab3a' | 'tab3b' | 'tab4'
    title: str
    body: str  # 1~2 sentence Korean summary
    severity: Literal["info", "warning", "critical"] = "info"


@dataclass(frozen=True)
class ExecutiveReport:
    """Full daily report. Maps to the AI 인사이트 카드 region."""

    generated_at: str
    headline: str  # 1 sentence top-line
    summary: str  # 5~10 sentences
    actions: list[str]  # 3 concrete recommendations
    cards: list[InsightCard]


# ─────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────
def generate_report(snapshot: AnalysisSnapshot) -> ExecutiveReport:
    """Generate the full executive report. Resilient to LLM failures.

    Flow:
      1) Build prompt from snapshot
      2) Call LLM (with retry + timeout)
      3) Parse strict-JSON response
      4) On any failure, fall back to template-based report
    """
    try:
        raw = _call_llm(_build_prompt(snapshot))
        return _parse_llm_response(raw, snapshot.generated_at)
    except Exception as exc:
        log.warning("LLM call failed (%s), falling back to template report", exc)
        return _template_report(snapshot)


def save_report(report: ExecutiveReport, path: Path = config.PART6_REPORT_JSON) -> None:
    """Save report as JSON for FastAPI to serve."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": report.generated_at,
        "headline": report.headline,
        "summary": report.summary,
        "actions": report.actions,
        "cards": [asdict(c) for c in report.cards],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("saved LLM report: %s", path)


# ─────────────────────────────────────────────────────────
# Snapshot builder — collects data from upstream parts
# ─────────────────────────────────────────────────────────
def build_snapshot_from_files() -> AnalysisSnapshot:
    """Read all upstream Part outputs from disk and bundle into a snapshot.

    실제로 파이프라인이 생성하는 파일들에서 직접 데이터를 빌드.
    대시보드가 사용하는 동일한 CSV들을 읽어 LLM 스냅샷을 구성한다.
    """
    generated_at = datetime.now(timezone.utc).isoformat()

    part1_records = _build_part1_route_aggregate()
    part2_summary = _summarize_part2()
    part3a_metrics = _build_part3a_metrics()
    part3b_records = _build_part3b_forecast()
    part5_records = _build_part5_shap()

    return AnalysisSnapshot(
        generated_at=generated_at,
        part1_route_aggregate=part1_records,
        part2_pattern_summary=part2_summary,
        part3a_regression_metrics=part3a_metrics,
        part3b_prophet_forecast=part3b_records,
        part5_shap_importance=part5_records,
    )


def _build_part1_route_aggregate(limit: int = 10) -> list[dict]:
    """ktx_long.csv에서 노선별 평균 공실률·관측치 수를 집계."""
    long_df = _read_optional_csv(config.LONG_CSV)
    if long_df is None or long_df.empty or config.ROUTE_COL not in long_df.columns:
        return []
    if config.TARGET_COL not in long_df.columns:
        return []
    agg = (
        long_df.groupby(config.ROUTE_COL, as_index=False)
        .agg(mean_vacancy=(config.TARGET_COL, "mean"),
             min_vacancy=(config.TARGET_COL, "min"),
             max_vacancy=(config.TARGET_COL, "max"),
             rows=(config.TARGET_COL, "count"))
    )
    agg["mean_vacancy"] = agg["mean_vacancy"].round(2)
    agg["min_vacancy"] = agg["min_vacancy"].round(2)
    agg["max_vacancy"] = agg["max_vacancy"].round(2)
    agg = agg.sort_values("mean_vacancy", ascending=False)
    return _df_to_records(agg, limit=limit)


def _build_part3a_metrics() -> dict | None:
    """model_comparison_per_route.csv에서 회귀 모델 성능 요약."""
    mc = _read_optional_csv(config.PROCESSED_DIR / "model_comparison_per_route.csv")
    if mc is None or mc.empty:
        return None

    def _parse_unified(val) -> float | None:
        import re
        m = re.search(r"-?\d+(\.\d+)?", str(val or ""))
        return float(m.group(0)) if m else None

    unified_mae = None
    if "통합모델_MAE" in mc.columns:
        unified_mae = _parse_unified(mc.iloc[0]["통합모델_MAE"])

    per_route = []
    for _, r in mc.iterrows():
        per_route.append({
            "노선": r.get("노선"),
            "개별모델_MAE": float(r.get("개별모델_MAE", 0) or 0),
            "개별모델_RMSE": float(r.get("개별모델_RMSE", 0) or 0),
            "데이터수": int(r.get("데이터수(행)", 0) or 0),
        })
    avg_mae = round(sum(r["개별모델_MAE"] for r in per_route) / max(len(per_route), 1), 4)
    avg_rmse = round(sum(r["개별모델_RMSE"] for r in per_route) / max(len(per_route), 1), 4)

    return {
        "통합모델_MAE": unified_mae,
        "개별모델_평균_MAE": avg_mae,
        "개별모델_평균_RMSE": avg_rmse,
        "노선별": per_route,
    }


def _build_part3b_forecast(limit: int = 12) -> list[dict]:
    """prophet 예측: 향후 12개월 샘플. 미래가 없으면 최신 12개."""
    fc = _read_optional_csv(config.PROPHET_FORECAST_CSV)
    if fc is None or fc.empty:
        return []
    if "ds" in fc.columns:
        fc = fc.copy()
        fc["ds"] = pd.to_datetime(fc["ds"], errors="coerce")
        future = fc[fc["ds"] >= pd.Timestamp.now()]
        sample = future.head(limit) if not future.empty else fc.tail(limit)
    else:
        sample = fc.tail(limit)
    return _df_to_records(sample, limit=limit)


def _build_part5_shap(limit: int = 10) -> list[dict]:
    """shap_per_route.csv 에서 변수별 평균 SHAP 중요도 집계 (전체 노선 평균)."""
    shap_df = _read_optional_csv(config.PROCESSED_DIR / "shap_per_route.csv")
    if shap_df is None or shap_df.empty:
        return []
    if "변수" not in shap_df.columns or "SHAP중요도" not in shap_df.columns:
        return []
    agg = (
        shap_df.groupby("변수", as_index=False)
        .agg(mean_abs_shap=("SHAP중요도", "mean"))
    )
    agg["mean_abs_shap"] = agg["mean_abs_shap"].round(4)
    agg = agg.sort_values("mean_abs_shap", ascending=False).rename(columns={"변수": "feature"})
    return _df_to_records(agg, limit=limit)


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        log.debug("optional file missing: %s", path)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        log.warning("failed to read %s: %s", path, exc)
        return None


def _df_to_records(df: pd.DataFrame | None, limit: int = 10) -> list[dict]:
    if df is None or df.empty:
        return []
    return df.head(limit).to_dict(orient="records")


def _summarize_part2() -> dict | None:
    """ktx_long.csv에서 시간·계절·명절 패턴을 요약."""
    long_df = _read_optional_csv(config.LONG_CSV)
    if long_df is None or long_df.empty or config.TARGET_COL not in long_df.columns:
        return None

    out: dict = {}

    # 월별 평균 공실률
    if "월" in long_df.columns:
        by_month = (
            long_df.groupby("월", as_index=False)[config.TARGET_COL]
            .mean()
            .round(2)
            .rename(columns={config.TARGET_COL: "평균공실률"})
        )
        out["월별_평균공실률"] = by_month.to_dict(orient="records")

    # 계절별 평균 공실률
    if "계절" in long_df.columns:
        by_season = (
            long_df.groupby("계절", as_index=False)[config.TARGET_COL]
            .mean()
            .round(2)
            .rename(columns={config.TARGET_COL: "평균공실률"})
        )
        out["계절별_평균공실률"] = by_season.to_dict(orient="records")

    # 최근 12개월 평균 / 초과수요 월 수
    if "yearmonth" in long_df.columns:
        long_df = long_df.copy()
        long_df["yearmonth"] = pd.to_datetime(long_df["yearmonth"], errors="coerce")
        recent = long_df.sort_values("yearmonth").tail(60)  # 노선 × 12개월
        if not recent.empty:
            out["최근평균공실률"] = round(float(recent[config.TARGET_COL].mean()), 2)
            out["최근초과수요월수"] = int((recent[config.TARGET_COL] < 0).sum())

    # 명절 효과 검정
    holiday = _read_optional_csv(config.PROCESSED_DIR / "holiday_mannwhitney.csv")
    if holiday is not None and not holiday.empty:
        out["명절효과"] = holiday.to_dict(orient="records")

    return out or None


# ─────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
당신은 한국철도공사 운영 데이터 분석 전문가입니다.
KTX 좌석 공실률·수요 예측 결과를 받아 운영자가 즉시 의사결정에 활용할 수 있는
한국어 인사이트를 작성합니다.

작성 원칙:
- 모든 수치는 입력 데이터에 명시된 값만 사용. 추정·예측은 절대 금지.
- 운영자에게 직접적으로 도움이 되는 액션 중심으로 작성.
- 격식체(~합니다)로 통일.
- 응답은 반드시 유효한 JSON. 추가 설명·마크다운 금지.
"""

_USER_PROMPT_TEMPLATE = """\
다음은 KTX 운영 분석 결과입니다.

[1] 노선별 공실률 집계 (상위 10건):
{part1}

[2] 시간·요일·계절 패턴 요약:
{part2}

[3-A] 회귀 예측 모델 성능:
{part3a}

[3-B] Prophet 시계열 예측 (향후 12개월, 상위 12건):
{part3b}

[5] SHAP 변수 영향도 (상위 10개):
{part5}

위 데이터를 바탕으로 다음 JSON 스키마에 맞춰 응답하세요.

{{
  "headline": "한 줄 요약 (15단어 이내)",
  "summary": "5~10문장 요약. 핵심 수치·노선·시점 포함.",
  "actions": ["권고 액션 1", "권고 액션 2", "권고 액션 3"],
  "cards": [
    {{"tab_id": "tab1", "title": "...", "body": "1~2문장", "severity": "info|warning|critical"}},
    {{"tab_id": "tab2", "title": "...", "body": "...", "severity": "..."}},
    {{"tab_id": "tab3a", "title": "...", "body": "...", "severity": "..."}},
    {{"tab_id": "tab3b", "title": "...", "body": "...", "severity": "..."}},
    {{"tab_id": "tab4", "title": "...", "body": "...", "severity": "..."}}
  ]
}}

severity 기준: info=일반, warning=공실률 30% 이상 또는 명절 임박, critical=공실률 40% 이상 또는 모델 MAPE 15% 초과
"""


def _build_prompt(snapshot: AnalysisSnapshot) -> str:
    """Render the user prompt with snapshot data inlined as JSON blocks."""
    return _USER_PROMPT_TEMPLATE.format(
        part1=_to_json_block(snapshot.part1_route_aggregate),
        part2=_to_json_block(snapshot.part2_pattern_summary),
        part3a=_to_json_block(snapshot.part3a_regression_metrics),
        part3b=_to_json_block(snapshot.part3b_prophet_forecast),
        part5=_to_json_block(snapshot.part5_shap_importance),
    )


def _to_json_block(payload: Any) -> str:
    """Compact JSON for prompt inlining."""
    if payload is None or (isinstance(payload, list) and not payload):
        return "(데이터 없음)"
    return json.dumps(payload, ensure_ascii=False, default=str)


# ─────────────────────────────────────────────────────────
# LLM client — provider-agnostic
# ─────────────────────────────────────────────────────────
def _call_llm(prompt: str, cfg: config.LLMConfig = config.LLMConfig()) -> str:
    """Dispatch to OpenAI GPT with retries."""
    return _call_openai(prompt, cfg)


def _call_openai(prompt: str, cfg: config.LLMConfig, retries: int = 2) -> str:
    """Call OpenAI Chat Completions with retries."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("install openai: pip install openai") from exc

    api_key = os.getenv(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"missing env var: {cfg.api_key_env}")

    client = OpenAI(api_key=api_key, timeout=cfg.timeout_seconds)

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            log.warning("OpenAI call failed (attempt %d): %s. retry in %ds", attempt + 1, exc, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


# ─────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────
def _parse_llm_response(raw: str, generated_at: str) -> ExecutiveReport:
    """Parse the JSON response. Strict but forgiving of code-fence wrappers."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").lstrip("json").strip()

    data = json.loads(cleaned)

    cards = [
        InsightCard(
            tab_id=c["tab_id"],
            title=c["title"],
            body=c["body"],
            severity=c.get("severity", "info"),
        )
        for c in data.get("cards", [])
    ]
    return ExecutiveReport(
        generated_at=generated_at,
        headline=data.get("headline", ""),
        summary=data.get("summary", ""),
        actions=list(data.get("actions", [])),
        cards=cards,
    )


# ─────────────────────────────────────────────────────────
# Fallback: template-based report (no LLM)
# ─────────────────────────────────────────────────────────
def _template_report(snapshot: AnalysisSnapshot) -> ExecutiveReport:
    """Deterministic template fallback used when the LLM API is unavailable.

    Ensures the dashboard *always* has insight cards to display.
    """
    headline = "AI 인사이트 생성 일시 불가 — 통계 요약을 대신 표시합니다"
    summary = (
        f"본 보고서는 {snapshot.generated_at} 시점에 생성되었습니다. "
        "LLM API 연결이 일시 불가하여 통계 기반 요약으로 대체합니다. "
        "환경설정에서 API 키를 확인한 후 다음 갱신 주기에 자동 복구됩니다."
    )
    actions = [
        "API 키 환경변수 확인 (ANTHROPIC_API_KEY 또는 OPENAI_API_KEY)",
        "재실행: python -m src.llm_report",
        "지속 실패 시 운영자에게 알림 발송",
    ]

    cards = [
        InsightCard("tab1", "노선별 공실률 현황", "최신 집계 데이터를 확인하세요.", "info"),
        InsightCard("tab2", "시간·요일·계절 패턴", "Heatmap에서 고공실 시간대를 확인하세요.", "info"),
        InsightCard("tab3a", "회귀 예측", "최신 모델 메트릭을 확인하세요.", "info"),
        InsightCard("tab3b", "시계열 예측", "향후 12개월 예측 추이를 검토하세요.", "info"),
        InsightCard("tab4", "변수 영향도", "SHAP 차트로 주요 변수를 확인하세요.", "info"),
    ]

    return ExecutiveReport(
        generated_at=snapshot.generated_at,
        headline=headline,
        summary=summary,
        actions=actions,
        cards=cards,
    )


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
def main() -> None:
    """CLI: build snapshot from upstream outputs, call LLM, save report."""
    config.ensure_directories()
    snapshot = build_snapshot_from_files()
    log.info(
        "snapshot built: part1=%d rows, part3b=%d rows, part5=%d rows",
        len(snapshot.part1_route_aggregate),
        len(snapshot.part3b_prophet_forecast),
        len(snapshot.part5_shap_importance),
    )

    report = generate_report(snapshot)
    save_report(report)

    log.info("headline: %s", report.headline)
    log.info("actions: %d items", len(report.actions))
    log.info("cards: %d items", len(report.cards))


if __name__ == "__main__":
    main()
