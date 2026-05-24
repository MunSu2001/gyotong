"""
KTX 공실률 분석 - 데이터 전처리 스크립트
입력: 4_고속철도_여객_수송동향.xlsx
출력: ktx_wide.csv, ktx_long.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────
INPUT_PATH = "4_고속철도_여객_수송동향.xlsx"
OUTPUT_WIDE = "ktx_wide.csv" # Wide 형태로 저장 (날짜x지표)
OUTPUT_LONG = "ktx_long.csv" # Long 형태로 저장 (노선별 행으로 변환)


def load_raw(path: str) -> pd.DataFrame:
    """엑셀 파일에서 Wide DataFrame 생성"""
    from openpyxl import load_workbook # 엑셀 읽기 라이브러리

    wb = load_workbook(path, data_only=True) # 데이터만 읽기(공식 계산값 반환)
    ws = wb.active # 첫 번째 시트 선택

    # 날짜 추출 (행3, 컬럼2~)
    dates, date_cols = [], []
    for c in range(2, ws.max_column + 1): # 컬럼 2번부터 끝까지 
        val = ws.cell(row=3, column=c).value # 행3의 값 읽기 
        if val and str(val).strip(): # 값이 있고 공백이 아니면
            dates.append(str(val).strip()) # 날짜 목록에 추가
            # 나중에 데이터를 읽을 때 "어느 컬럼에 데이터가 있나"를 빠르게 접근하기 위해 인덱스 저장
            date_cols.append(c) # 컬럼 번호도 기억해두기
            
    # 지표명 추출 (컬럼1, 행4~)
    indicators, indicator_rows = [], []
    for r in range(4, ws.max_row + 1): # 행4부터 끝까지
        val = ws.cell(row=r, column=1).value # 컬럼 1번(A)의 값
        if val and str(val).strip() not in ["", "출처:"]: # 빈칸과 출처는 제외
            indicators.append(str(val).strip().replace("\xa0", " ").strip())
            indicator_rows.append(r)

    # 데이터 추출
    data = {}
    for ind, r in zip(indicators, indicator_rows): # 지표명과 행 번호 쌍으로 반복
        row_data = []
        for c in date_cols: # 각 날짜 컬럼에서
            val = ws.cell(row=r, column=c).value
            if val == "-" or val is None: # "-" 또는 빈값 -> nan
                row_data.append(np.nan)
            else:
                # 엑셀에 예기치 않은 텍스트가 있을 수 있으므로 try-except 숫자 변환
                try:
                    row_data.append(float(str(val).replace(",", ""))) # 숫자로 변환
                except Exception:
                    row_data.append(np.nan) # 변환 실패 -> nan
        data[ind] = row_data # {지표명 : [데이터리스트]}

    # ---DataFrame 생성---
    df = pd.DataFrame(data, index=dates)
    df.index = pd.to_datetime(
        df.index.str.replace("월", ""), format="%Y%m" # "200404월" -> "200404" -> datetime(2004, 4, 1)
    )
    df.index.name = "yearmonth" # 인덱스 이름 설정
    return df


def add_vacancy_rate(df: pd.DataFrame) -> pd.DataFrame:
    """공실률 = 100 - 이용률 컬럼 추가"""
    # 엑셀 원본의 컬럼명이 불규칙 -> mapping 사용
    mapping = {
        "KTX이용률":       "KTX_공실률", # 전체 평균
        "경부선 KTX이용률": "경부선_KTX_공실률", 
        "호남선 KTX이용률": "호남선_KTX_공실률",
        "경전선  KTX이용률":"경전선_KTX_공실률", # 주의: 컬럼명에 공백 2개
        "전라선 KTX이용률": "전라선_KTX_공실률",
        "동해선 KTX이용률": "동해선_KTX_공실률",
    }
    for src, dst in mapping.items():
        if src in df.columns: # 해당 컬럼이 있으면
            df[dst] = 100 - df[src] # 공실률 = 100 - 이용률
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """날짜 파생변수 추가"""
    # 분리하는 이유 -> 모델이 "4월이면 공실률이 높다" 같은 패턴을 학습하려면 월/분기/계절이 별도 컬럼으로 있어야 함
    df["연도"] = df.index.year # 인덱스에서 연도 추출
    df["월"] = df.index.month # 인덱스에서 월 추출
    df["분기"] = df.index.quarter # 인덱스에서 분기 추출
    df["계절"] = df["월"].map({
        12: "겨울", 1: "겨울", 2: "겨울",
        3: "봄",   4: "봄",   5: "봄",
        6: "여름", 7: "여름", 8: "여름",
        9: "가을", 10: "가을", 11: "가을",
    })
    # 이벤트 플래그
    df["코로나기간"] = ((df.index >= "2020-01") & (df.index <= "2022-06")).astype(int) # 코로나 기간 공실률이 높을 가능성 있음
    df["SRT개통후"] = (df.index >= "2016-12").astype(int) # SRT 개통 이후 공실률 패턴 변화 가능성 있음
    return df


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """노선별 Long 형태 변환""" # Long 형태 (한 행 = 한월, 여러 컬럼 = 여러 노선)
    노선_map = {
        "경부선": ("경부선 KTX여객수", "경부선 KTX이용률", "경부선_KTX_공실률"),
        "호남선": ("호남선 KTX여객수", "호남선 KTX이용률", "호남선_KTX_공실률"),
        "경전선": ("경전선 KTX여객수", "경전선  KTX이용률", "경전선_KTX_공실률"),
        "전라선": ("전라선 KTX여객수", "전라선 KTX이용률", "전라선_KTX_공실률"),
        "동해선": ("동해선 KTX여객수", "동해선 KTX이용률", "동해선_KTX_공실률"),
    }
    rows = []
    for idx, row in df.iterrows(): # Wide의 각 행(월) 반복
        for 노선, (여객col, 이용률col, 공실률col) in 노선_map.items(): # 각 노선 반복
            공실률 = row.get(공실률col, np.nan) # 공실률 읽기 
            if pd.isna(공실률):
                continue # 데이터 없으면 스킵

            rows.append({ 
                # 한 행 구성
                "yearmonth":      idx,
                "연도":           row["연도"],
                "월":             row["월"],
                "분기":           row["분기"],
                "계절":           row["계절"],
                "코로나기간":     row["코로나기간"],
                "SRT개통후":      row["SRT개통후"],
                "노선":           노선,
                "여객수_천명":    row.get(여객col, np.nan),
                "KTX이용률":      row.get(이용률col, np.nan),
                "공실률":         공실률,
                "초과수요":       1 if 공실률 < 0 else 0,
                "공실상태":       (
                    "초과수요"
                    if 공실률 < 0
                    else "여유없음"
                    if 공실률 < 10
                    else "공실있음"
                ),
                "KTX전체이용률":  row.get("KTX이용률", np.nan),
                "KTX전체공실률":  row.get("KTX_공실률", np.nan),
            })
    df_long = pd.DataFrame(rows)
    return df_long.reset_index(drop=True)


def main():
    print("1. 원본 데이터 로드...")
    df_wide = load_raw(INPUT_PATH)

    print("2. 공실률 컬럼 추가...")
    df_wide = add_vacancy_rate(df_wide)

    print("3. 시간 파생변수 추가...")
    df_wide = add_time_features(df_wide)

    print("4. Long 형태 변환...")
    df_long = to_long(df_wide)

    print("5. CSV 저장...")
    df_wide.to_csv(OUTPUT_WIDE, encoding="utf-8-sig")
    df_long.to_csv(OUTPUT_LONG, index=False, encoding="utf-8-sig")

    print(f"\n✅ 완료")
    print(f"   Wide: {df_wide.shape} → {OUTPUT_WIDE}")
    print(f"   Long: {df_long.shape} → {OUTPUT_LONG}")
    print(f"\n노선별 공실률 기초통계:")
    print(df_long.groupby("노선")["공실률"].describe().round(1).to_string())


if __name__ == "__main__":
    main()
