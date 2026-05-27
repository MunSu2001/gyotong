# VACANT — KTX 좌석 공실 분석 플랫폼

공공데이터 기반 KTX 좌석 공실률 분석 · 예측 · 운영 인사이트 자동 생성 시스템.

## 구조

```
src/
├─ config.py           모든 경로·상수·하이퍼파라미터 중앙화
├─ logging_setup.py    표준 로깅
├─ io_utils.py         공통 IO 헬퍼
├─ metrics.py          MAE/RMSE/MAPE 공용 계산
├─ preprocess.py       파트 0: 원본 데이터 정제
├─ features.py         파트 0: 피처 엔지니어링
├─ part1_aggregate.py  파트 1: 노선·역별 집계
├─ model.py            파트 3-A: CatBoost + LightGBM 회귀
├─ prophet_model.py    파트 4: Prophet 시계열 예측
├─ shap_analysis.py    파트 5: SHAP 변수 영향도
└─ llm_report.py       파트 6: LLM 자연어 인사이트 생성
api/
└─ main.py             FastAPI: 대시보드 백엔드
dashboard/             Next.js 프론트엔드
```

## 빠른 시작

### 1. 환경설정

```bash
# 라이브러리 설치
pip install -r requirements.txt

# 환경변수 파일 생성
cp .env.sample .env
# .env 파일을 열어서 OPENAI_API_KEY 입력
```

`.env` 파일 예시:
```
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```

`config.py`가 자동으로 `.env`를 로드합니다.

### 2. 파이프라인 실행

> 프로젝트 루트에서 
> `python -m src.model` 으로 실행

```bash
# 데이터 전처리 → 피처 → 집계 → 모델 → 예측 → SHAP → LLM 리포트
python -m src.preprocess
python -m src.features
python -m src.part1_aggregate
python -m src.model
python -m src.prophet_model
python -m src.shap_analysis
python -m src.llm_report
```

### 3. API 서버 실행

```bash
uvicorn api.main:app --reload --port 8000
```

API 문서: `http://localhost:8000/docs`

### 4. 대시보드 실행

```bash
cd dashboard
npm install
npm run dev -- -p 3001
```

## 주요 엔드포인트

| Method | Path | 용도 |
|--------|------|------|
| GET | `/health` | liveness probe |
| GET | `/api/v1/aggregate` | 파트 1 — 노선별 공실 집계 |
| GET | `/api/v1/forecast?route=경부선` | 파트 4 — 시계열 예측 |
| GET | `/api/v1/shap` | 파트 5 — SHAP 변수 영향도 |
| GET | `/api/v1/report` | 파트 6 — LLM 캐시 리포트 |
| POST | `/api/v1/report/refresh` | 파트 6 — 리포트 강제 갱신 |

## LLM 리포트 구성

`llm_report.py`는 파트 1~5의 출력을 자동 수집해 다음을 생성:

- **headline**: 한 줄 요약 (대시보드 상단)
- **summary**: 5~10문장 요약
- **actions**: 권고 액션 3개
- **cards**: 탭별 인사이트 카드 5개

API 키 미설정 시 자동으로 템플릿 폴백 → 대시보드는 항상 작동.

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `LLM_MODEL` | `gpt-4o` | OpenAI 모델명 (`gpt-4o-mini`도 가능) |
| `OPENAI_API_KEY` | (필수) | OpenAI API 키 |
