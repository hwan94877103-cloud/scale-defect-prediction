"""
후판공정 Scale(산화철) 불량 예측 — 전처리 및 Feature Set 설계 모듈
================================================================

후판(Heavy Plate) 공정에서 발생하는 Scale 불량을 예측하기 위한 전처리 파이프라인.
원본 CSV 적재부터 도메인 파생변수 생성, 데이터 누수 방지를 위한 Feature Set 분리,
학습/평가 분할까지를 담당한다.

공정 흐름
---------
    가열로(예열대 → 가열대 → 균열대) → 추출 → HSB(고압수 세척) → 압연 → Scale 판정

스케일 생성 원리
---------------
강재를 1,100~1,200°C로 가열하면 표면 철(Fe)이 산소·수증기와 반응해 산화철 껍질이
생긴다. 층 구조는 안쪽부터 FeO(뷔스타이트) → Fe3O4(마그네타이트) → Fe2O3(헤마타이트).

불량을 가르는 것은 "스케일이 생겼느냐"가 아니라 "HSB로 떨어지느냐"이다.
스케일은 반드시 생기며, 잘 떨어지면 양품이다. 떨어지지 않는 조건은 셋이다.

    1. 너무 두꺼움   : 고온·장시간 노출            → 가열로 변수군
    2. 너무 치밀함   : 고온 유지로 조직이 여물어 밀착 → 균열대 변수군
    3. 화학적 밀착   : Si 함량이 높으면 페이얄라이트(Fe2SiO4, 융점 약 1,173°C)가
                      생성되어 액상이 지철 계면에 침투 → 강종 변수군

사용 예시
--------
    >>> from preprocessing import build_dataset, split_dataset
    >>> df = build_dataset("data/SCALE불량.csv")
    >>> x_tr, x_te, y_tr, y_te, prep = split_dataset(df, feature_set="A")
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = Pipeline([("prep", prep), ("clf", RandomForestClassifier())])
    >>> model.fit(x_tr, y_tr)

Author  : (your name)
License : MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

__all__ = [
    "load_raw",
    "clean_data",
    "add_domain_features",
    "engineer_one_row",
    "build_dataset",
    "get_feature_columns",
    "build_preprocessor",
    "split_dataset",
    "FEATURE_SETS",
    "SplitStrategy",
]


# =============================================================================
# 상수 정의
# =============================================================================

DEFAULT_ENCODING = "cp949"          # 원본이 한글 Windows 환경에서 저장되어 UTF-8 아님
DEFAULT_TEST_SIZE = 0.25
DEFAULT_SEED = 42

COL_DATE = "rolling_date"           # 시간 기준 분할의 정렬 키
COL_TARGET_RAW = "scale"            # 원본 타깃 (범주형: '양품' / '불량')
COL_TARGET = "target"               # 이진화 타깃 (불량=1, 양품=0)
LABEL_DEFECT = "불량"

# 페이얄라이트(Fe2SiO4) 융점 [°C].
# 균열대 온도가 이 값을 넘으면 저융점 액상 스케일이 지철 계면에 침투해
# HSB(고압수)로도 떼어내기 힘든 '난제거성 스케일'이 된다.
FAYALITE_MELTING_POINT = 1173

# 원본 CSV가 반드시 가지고 있어야 하는 컬럼
REQUIRED_COLUMNS = [
    "plate_no", "rolling_date", "scale", "spec_long", "spec_country",
    "steel_kind", "pt_thick", "pt_width", "pt_length", "hsb",
    "fur_no", "fur_input_row", "fur_heat_temp", "fur_heat_time",
    "fur_soak_temp", "fur_soak_time", "fur_total_time",
    "rolling_method", "rolling_temp", "descaling_count", "work_group",
]

# -----------------------------------------------------------------------------
# NOTE: 추출대 온도(fur_ex_temp)에 대하여
#
# 일반적인 후판 도메인 문헌에서는 '추출대 온도(가열로를 빠져나와 압연기로 진입하기
# 직전의 표면 온도)'를 스케일 불량의 최종 방아쇠로 본다. 그러나 본 데이터셋에는
# 해당 컬럼이 존재하지 않는다.
#
# 그 역할은 rolling_temp(압연 온도)가 대신 수행하며, 실제로 양품/불량 간
# 평균 격차가 전체 변수 중 가장 크다 (불량 988.8°C vs 양품 910.3°C, +78.5°C).
# 원 데이터에서 fur_ex_temp를 추가 확보할 수 있다면 예측력 향상이 기대된다.
# -----------------------------------------------------------------------------


# =============================================================================
# Feature Set 정의  ★ 본 모듈의 핵심 설계
# =============================================================================
#
# 데이터 누수(Data Leakage) 방지를 위해 변수를 '예측 시점' 기준으로 분리한다.
#
# rolling_temp, descaling_count, rolling_method는 압연이 끝나야 확정되는 값이다.
# 그런데 Scale 불량 판정 역시 압연 후에 이뤄진다. 즉 이 변수들을 예측에 사용하면
# "결과를 보고 결과를 맞히는" 구조가 되어, 정확도는 높지만 현업에서 조치할 시점이
# 이미 지나 있다.
#
# 실측 결과 (RandomForest, Permutation Importance)
# ------------------------------------------------
#   Set_B(전공정) : Accuracy 0.976 — 단 rolling_temp가 중요도 0.46으로 독식하여
#                   21개 변수 중 실질적으로 4개만 기여. XAI 해석 여지가 좁다.
#   Set_A(사전예측): Accuracy 0.936 — 성능은 4%p 낮지만 중요도가 8개 변수로 분산되고
#                   (pt_width, steel_kind, hsb, sa_ratio, fur_soak_temp ...),
#                   전부 가열로 추출 이전에 확정되는 값이라 실시간 경보에 사용 가능.
#
# =============================================================================

# --- 상류(Upstream) : 가열로 추출 시점에 이미 확정된 변수 --------------------

CAT_UPSTREAM = [
    "spec_country",     # 규격 국가. 국가 자체가 원인이 아니라 강종 조성의 프록시다.
                        # 일본 규격재 불량률 51.9% vs 공통 8.3%.
    "steel_kind",       # 강 종류 (C=탄소강 추정 / T=TMCP 지정재 추정).
                        # C 38.4% vs T 8.2%로 약 5배 차이. 정의는 현장 확인 필요.
    "grade",            # spec_long에서 추출한 강도 등급 (A / AH32 / EH36 ...).
    "fur_no",           # 가열로 호기(1~3). 버너 노후도, 공기비(λ) 제어 상태,
                        # 내화물 열화에 따라 같은 설정온도라도 실제 산화량이 다르다.
                        # 유의차가 있으면 설비 점검 대상을 특정할 수 있다.
    "fur_input_row",    # 장입 열(1열/2열). 버너 화염과의 거리, 배기 흐름이 달라
                        # 실제 체감 온도가 다르다. 스키드 빔 접촉부는 국부적으로
                        # 온도가 낮아 스케일 두께 편차(skid mark)를 만든다.
    "work_group",       # 작업조(1~4조). 장입 간격 판단, 추출 타이밍 등 운전 습관 차이.
                        # 단, 조별 담당 강종·시간대가 다르므로 교란변수 통제 후 해석할 것.
    "hsb",              # 고압수 디스케일러(High-pressure Scale Breaker) 적용 여부.
                        # 150~250bar 고압수로 스케일을 박리시키는, 스케일 불량의
                        # 최후이자 유일한 물리적 방어선.
                        # ★ 미적용 47건이 예외 없이 100% 불량 → 완전분리(perfect separation).
                        #   단, 이는 전체 불량 310건의 15.2%에 불과하므로 나머지 263건은
                        #   다른 변수가 설명해야 한다. 실측 중요도에서도 hsb는 1위가 아닌
                        #   2위였다. 따라서 강력한 도메인 팩트로 그대로 포함시킨다.
]

NUM_UPSTREAM = [
    # --- 제품 치수 : 열용량 = 필요 재로시간 ---
    "pt_thick",         # 두께(mm). 두꺼우면 재로시간이 길어져 스케일이 두꺼워지고,
                        # 얇으면 압연 중 온도 강하가 빨라 패스 수가 늘면서 잔류
                        # 스케일이 롤에 눌려 압입(rolled-in scale)된다. 양방향 위험.
    "pt_width",         # 폭(mm). HSB 노즐 헤더는 고정폭이라 판이 넓으면 에지부
                        # 충돌압력이 떨어져 스케일이 남는다. 실측 Set_A 중요도 1위.
    "pt_length",        # 길이(mm). 길수록 판 앞뒤(top/bottom) 압연 온도 편차가 커진다.
    "sa_ratio",         # 비표면적 = 표면적/체적. 단위 질량당 산화 노출 정도.
                        # 스케일 분석에서 물리적으로 가장 타당한 파생변수.
    "aspect",           # 세장비 = 길이/폭. 압연 방향 온도 편차의 프록시.
    "is_tm",            # TMCP 지정재 여부(spec_long의 '-TM' 표기).

    # --- 가열로 : 스케일 '생성' 단계 ---
    "fur_heat_temp",    # 가열대 온도(°C). 철판 내외부 온도차가 가장 큰 구간으로,
                        # 지나치게 높으면 급격한 열충격과 함께 표면 산화가 격렬히
                        # 시작되는 '불량의 도화선'. 불량 1164.3 vs 양품 1154.1 (+10.2).
    "fur_heat_time",    # 가열대 체류시간(분). 단독보다 온도와의 곱으로 봐야 한다.
    "fur_soak_temp",    # 균열대 온도(°C). 압연 직전 단계라 이 온도가 높으면 표면
                        # 산화철이 튼실하고 단단하게 '여물어' HSB로도 안 떨어진다.
                        # 불량 1159.4 vs 양품 1147.1 (+12.3) — 단일 온도변수 중 최대 격차.
    "fur_soak_time",    # 균열대 재로시간(분). 온도가 높고 이 시간이 길수록 스케일이
                        # 두껍고 단단하게 여문다. 단 실데이터는 불량 쪽이 오히려 짧은데,
                        # 산화속도가 온도에 지수적(아레니우스)·시간에 선형이므로
                        # "고온·단시간"이 "저온·장시간"보다 위험할 수 있다.
    "fur_total_time",   # 총 재로시간(분). 생산 지연으로 길어지면 산화철이 걷잡을 수
                        # 없이 두꺼워져 HSB로도 제거가 안 되는 상황이 발생한다.
    "fur_wait_time",    # 대기시간 = 총재로 - 가열 - 균열. 예열대 체류 + 순수 지연분.
    "heat_index",       # 가열 열량지수 = 가열대온도 × 가열시간 / 1000.
    "soak_index",       # 균열 열량지수 = 균열대온도 × 균열시간 / 1000.
    "soak_over_1173",   # 페이얄라이트 임계 플래그 (균열대 온도 ≥ 1173°C).

    # --- 운전 컨텍스트 ---
    "work_hour",        # 압연 시각(0~23). 야간·교대 인수인계 시점의 프록시.
    "gap_min",          # 직전 판과의 압연 간격(분). 간격이 벌어졌다는 것은 후공정
                        # 지연으로 앞의 판들이 가열로에서 대기했다는 뜻이며,
                        # fur_total_time이 늘어난 '원인'을 설명해준다.
]

# --- 하류(Downstream) : 압연이 끝나야 확정되는 변수 --------------------------

CAT_DOWNSTREAM = [
    "rolling_method",   # 압연 방식. TMCP는 가속냉각을 동반해 압연온도를 의도적으로
                        # 낮게 관리하므로 2차 스케일 성장이 억제된다.
                        # CR 35.4% vs TMCP 8.1%.
]

NUM_DOWNSTREAM = [
    "rolling_temp",     # 압연 온도(°C). 가열로에서 아무리 온도를 잘 잡아도 이 온도가
                        # 높으면 압연 직전 스케일이 마지막으로 급격히 성장·치밀화되어
                        # 난제거성이 된다. '불량 발생의 최종 방아쇠'.
                        # 불량 988.8 vs 양품 910.3 (+78.5) — 전체 변수 중 최대 격차.
    "descaling_count",  # 디스케일링 횟수(5~10). 많을수록 2차 스케일 제거에 유리하나
                        # 분사할 때마다 판 온도가 떨어져 압연 하중이 올라간다.
                        # ⚠ 역인과 의심: 불량률이 횟수에 대해 단조롭지 않다
                        #   (5회 100%, 6회 13.6%, 7회 100%, 8회 49%, 9회 100%, 10회 21.1%).
                        #   스케일이 안 떨어지는 것을 보고 오퍼레이터가 추가 분사한
                        #   결과라면 원인이 아니라 '결과의 대리 지표'이며, 모델에 넣으면
                        #   예측 시점에 알 수 없는 정보를 쓰는 리키지가 된다.
                        #   현장에 "사전 설정값인가, 작업 중 판단값인가" 확인 필요.
]


FEATURE_SETS: dict[str, dict] = {
    # 사전 예측용 — 실시간 경보 및 오퍼레이터 지원
    # 가열로에서 슬래브가 나오는 시점에 이미 알 수 있는 값만 사용하므로,
    # "이 판은 위험합니다" 경보를 띄우면 압연온도를 낮추거나 디스케일링을
    # 추가하는 등 실제 조치가 가능하다.
    "A": {
        "cat": CAT_UPSTREAM,
        "num": NUM_UPSTREAM,
        "desc": "사전 예측용 (가열로 추출 시점 변수만)",
    },
    # 사후 분석용 — 근본 원인 규명 및 XAI
    # 전 공정 변수를 사용해 "무엇이 불량을 만들었나"를 설명한다.
    # 예측 성능은 높지만 실시간 경보에는 사용할 수 없다.
    "B": {
        "cat": CAT_UPSTREAM + CAT_DOWNSTREAM,
        "num": NUM_UPSTREAM + NUM_DOWNSTREAM,
        "desc": "사후 분석용 (압연 공정 변수 포함)",
    },
}

FeatureSetName = Literal["A", "B"]
SplitStrategy = Literal["time", "random"]


# =============================================================================
# 1) 원본 적재
# =============================================================================

def load_raw(path: str | Path, encoding: str = DEFAULT_ENCODING) -> pd.DataFrame:
    """원본 CSV를 적재한다.

    원본은 한글 Windows 환경에서 저장되어 UTF-8이 아니므로 cp949로 읽어야 한다.
    UTF-8로 읽으면 UnicodeDecodeError가 발생한다.

    Parameters
    ----------
    path : str | Path
        CSV 파일 경로.
    encoding : str, default "cp949"
        문자 인코딩. euc-kr로도 동작한다.

    Returns
    -------
    pd.DataFrame
        원본 데이터프레임 (1,000행 × 21열).

    Raises
    ------
    FileNotFoundError
        파일이 존재하지 않을 때.
    ValueError
        필수 컬럼이 누락되었을 때.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")

    df_raw = pd.read_csv(path, encoding=encoding)

    missing = set(REQUIRED_COLUMNS) - set(df_raw.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 누락되었습니다: {sorted(missing)}")

    return df_raw


# =============================================================================
# 2) 정제 — 결측·이상치 처리, 타입 변환, 타깃 이진화
# =============================================================================

def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """결측·이상치를 정제하고 타입을 변환한다.

    수행 내용
    ---------
    1. rolling_temp == 0 → NaN
       상온 압연은 물리적으로 불가능하다. 이 0은 온도계 미측정 또는 통신 두절을
       0으로 코딩한 결측값이다(총 6건, 전부 양품). 그대로 두면 평균이 아래로
       끌려가고 표준화가 왜곡된다. 실제 결측 대체는 학습 파이프라인 내부의
       SimpleImputer가 fold별로 수행하므로 여기서는 NaN 표시만 한다.
    2. rolling_date → datetime
       원본은 SAS 날짜형 문자열('03JAN2023:07:07:53')이다.
    3. scale → target 이진화
       Recall 우선 문제이므로 '불량'을 positive class(1)로 둔다.

    Parameters
    ----------
    df : pd.DataFrame
        load_raw()의 반환값.
    verbose : bool, default True
        정제 결과 요약 출력 여부.

    Returns
    -------
    pd.DataFrame
        정제된 사본 (원본은 변경하지 않음).
    """
    out = df.copy()

    # --- 2-1. rolling_temp 이상치 → 결측 ---
    n_zero = int((out["rolling_temp"] == 0).sum())
    out["rolling_temp"] = out["rolling_temp"].replace(0, np.nan)

    # --- 2-2. 날짜형 변환 (SAS 형식: DDMMMYYYY:HH:MM:SS) ---
    out["rolling_date"] = pd.to_datetime(
        out["rolling_date"], format="%d%b%Y:%H:%M:%S"
    )

    # --- 2-3. 타깃 이진화 (불량=1, 양품=0) ---
    out[COL_TARGET] = (out[COL_TARGET_RAW] == LABEL_DEFECT).astype(int)

    if verbose:
        print(f"[clean] rolling_temp == 0 → NaN : {n_zero}건")
        print(f"[clean] 결측 컬럼 : "
              f"{out.isna().sum()[lambda s: s > 0].to_dict()}")
        print(f"[clean] 불량률 : {out[COL_TARGET].mean() * 100:.1f}% "
              f"({out[COL_TARGET].sum()} / {len(out)})")

    return out


# =============================================================================
# 3) 도메인 파생변수
# =============================================================================

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """야금학적 도메인 지식에 기반한 파생변수를 생성한다.

    생성 변수
    ---------
    gap_min         직전 판과의 압연 간격(분) — 생산 지연의 프록시
    work_hour       압연 시각(0~23)
    surface_area    표면적 — 산화 노출 면적
    volume          체적 — 열용량
    sa_ratio        비표면적 = 표면적/체적 ★
    aspect          세장비 = 길이/폭
    fur_wait_time   대기시간 = 총재로 - 가열 - 균열
    heat_index      가열 열량지수 = 온도 × 시간 / 1000
    soak_index      균열 열량지수 = 온도 × 시간 / 1000
    soak_over_1173  페이얄라이트 임계 플래그
    is_tm           TMCP 지정재 여부
    grade           강도 등급 (spec_long에서 추출)

    Parameters
    ----------
    df : pd.DataFrame
        clean_data()의 반환값.

    Returns
    -------
    pd.DataFrame
        파생변수가 추가된 사본. rolling_date 오름차순으로 정렬된다.
    """
    out = df.copy().sort_values("rolling_date").reset_index(drop=True)

    # --- 3-1. 시간 파생 ---
    # 압연 간격이 벌어졌다는 것은 후공정 지연으로 앞의 판들이 가열로 안에서
    # 대기했다는 뜻이다. fur_total_time이 늘어난 '원인'을 설명하는 변수.
    out["gap_min"] = out["rolling_date"].diff().dt.total_seconds() / 60
    out["gap_min"] = out["gap_min"].fillna(out["gap_min"].median())
    out["work_hour"] = out["rolling_date"].dt.hour

    # --- 3-2. 치수 파생 ---
    thick, width, length = out["pt_thick"], out["pt_width"], out["pt_length"]
    out["surface_area"] = 2 * (thick * width + width * length + thick * length)
    out["volume"] = thick * width * length
    # 비표면적: 클수록 단위 질량당 산화 노출이 심하다.
    out["sa_ratio"] = out["surface_area"] / out["volume"]
    out["aspect"] = length / width

    # --- 3-3. 가열로 파생 ---
    # 총 재로시간에서 가열대·균열대 체류를 빼면 예열대 체류 + 순수 대기 지연이
    # 남는다. 생산 지연의 직접적인 영향을 분리해서 볼 수 있다.
    out["fur_wait_time"] = (
        out["fur_total_time"] - out["fur_heat_time"] - out["fur_soak_time"]
    )

    # 산화량은 온도와 시간의 곱에 지배되므로 교호항을 명시적으로 만들어준다.
    out["heat_index"] = out["fur_heat_temp"] * out["fur_heat_time"] / 1000
    out["soak_index"] = out["fur_soak_temp"] * out["fur_soak_time"] / 1000

    # 페이얄라이트 임계 플래그. 이 온도를 넘으면 저융점 액상 스케일이
    # 지철 계면에 침투해 HSB로도 제거가 어려워진다.
    out["soak_over_1173"] = (
        out["fur_soak_temp"] >= FAYALITE_MELTING_POINT
    ).astype(int)

    # --- 3-4. 규격 분해 ---
    # spec_long은 66종이라 원핫 인코딩하면 차원만 늘어난다.
    # 의미 있는 축으로 쪼개서 사용한다.
    out["is_tm"] = out["spec_long"].str.contains("-TM", na=False).astype(int)
    out["grade"] = (
        out["spec_long"].str.extract(r"([AEDH]H?\d{2})")[0].fillna("BASE")
    )

    return out


def engineer_one_row(inputs: dict) -> dict:
    """단일 레코드용 파생변수 계산 — 학습과 서빙이 공유한다.

    ★ 이 함수가 존재하는 이유 (train/serve skew 방지)

    대시보드는 사용자가 입력한 값 하나로 예측한다. 이때 파생변수를 앱 쪽에서
    따로 계산하면, 같은 계산식이 preprocessing.py와 app.py 두 곳에 존재하게
    된다. 나중에 한쪽만 수정하면 **에러 없이 조용히 틀린 예측**이 나온다.

        연습할 때 쓴 저울과 시합에서 쓴 저울이 다른 상황

    add_domain_features()와 이 함수는 동일한 식을 사용하며, 아래 테스트로
    일치를 검증한다.

        python preprocessing.py --data data/SCALE불량.csv

    Parameters
    ----------
    inputs : dict
        최소한 다음 키를 포함해야 한다.
        fur_heat_temp, fur_heat_time, fur_soak_temp, fur_soak_time,
        fur_total_time, pt_thick, pt_width, pt_length

    Returns
    -------
    dict
        입력 dict의 사본에 파생변수가 추가된 것.
    """
    out = dict(inputs)

    # 열량지수 : 산화량은 온도 × 시간의 곱에 지배됨
    out["heat_index"] = out["fur_heat_temp"] * out["fur_heat_time"] / 1000
    out["soak_index"] = out["fur_soak_temp"] * out["fur_soak_time"] / 1000

    # 페이얄라이트 임계 플래그
    out["soak_over_1173"] = int(out["fur_soak_temp"] >= FAYALITE_MELTING_POINT)

    # 대기시간 = 총재로 - 가열 - 균열
    out["fur_wait_time"] = (
        out["fur_total_time"] - out["fur_heat_time"] - out["fur_soak_time"]
    )

    # 치수 파생
    thick, width, length = out["pt_thick"], out["pt_width"], out["pt_length"]
    out["surface_area"] = 2 * (thick * width + width * length + thick * length)
    out["volume"] = thick * width * length
    out["sa_ratio"] = out["surface_area"] / out["volume"]
    out["aspect"] = length / width

    return out


# =============================================================================
# 4) 통합 진입점
# =============================================================================

def build_dataset(
    path: str | Path,
    encoding: str = DEFAULT_ENCODING,
    verbose: bool = True,
) -> pd.DataFrame:
    """적재 → 정제 → 파생변수 생성을 한 번에 수행한다.

    Parameters
    ----------
    path : str | Path
        원본 CSV 경로.
    encoding : str, default "cp949"
    verbose : bool, default True

    Returns
    -------
    pd.DataFrame
        분석 준비가 끝난 데이터프레임.
    """
    df = load_raw(path, encoding=encoding)
    if verbose:
        print(f"[load ] shape : {df.shape}")

    df = clean_data(df, verbose=verbose)
    df = add_domain_features(df)

    if verbose:
        print(f"[feat ] shape : {df.shape} (파생 {df.shape[1] - len(REQUIRED_COLUMNS) - 1}개 추가)")

    return df


# =============================================================================
# 5) Feature Set 조회 및 전처리기 구성
# =============================================================================

def get_feature_columns(feature_set: FeatureSetName = "A") -> tuple[list[str], list[str]]:
    """Feature Set 이름으로 범주형/수치형 컬럼 목록을 반환한다.

    Parameters
    ----------
    feature_set : {"A", "B"}
        A: 사전 예측용 (가열로 추출 시점 변수만)
        B: 사후 분석용 (압연 공정 변수 포함)

    Returns
    -------
    (cat_cols, num_cols) : tuple[list[str], list[str]]
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"feature_set은 {list(FEATURE_SETS)} 중 하나여야 합니다: {feature_set!r}"
        )
    spec = FEATURE_SETS[feature_set]
    return list(spec["cat"]), list(spec["num"])


def build_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    """전처리기를 구성한다.

    ★ 반드시 Pipeline 안에 넣어 사용할 것.

    전체 데이터에 먼저 fit_transform을 걸고 나서 train/test를 나누면
    테스트셋의 평균·표준편차가 학습에 새어 들어간다(표준화 단계 데이터 누수).
    ColumnTransformer를 Pipeline에 넣으면 교차검증 fold마다 학습 데이터만으로
    다시 fit되므로 이 문제가 구조적으로 차단된다.

    결측 대체(SimpleImputer) 역시 같은 이유로 파이프라인 내부에 둔다.
    rolling_temp의 결측 6건이 여기서 fold별 중앙값으로 채워진다.

    Parameters
    ----------
    cat_cols : list[str]
        범주형 컬럼.
    num_cols : list[str]
        수치형 컬럼.

    Returns
    -------
    ColumnTransformer
    """
    return ColumnTransformer([
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols,
        ),
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            num_cols,
        ),
    ])


# =============================================================================
# 6) 학습/평가 분할
# =============================================================================

def split_dataset(
    df: pd.DataFrame,
    feature_set: FeatureSetName = "A",
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    hsb_only: bool = False,
    strategy: SplitStrategy = "time",
    verbose: bool = True,
):
    """Feature Set에 맞춰 분할하고 전처리기를 함께 반환한다.

    분할 방식 (strategy)
    -------------------
    "time" (기본값)
        rolling_date 오름차순 정렬 후 앞을 학습, 뒤를 평가로 자른다.
        실제 조업은 "과거로 미래를 예측"하는 상황이므로 이 방식이 현실을 재현한다.

    "random"
        층화 랜덤 분할. 참고 비교용으로만 남겨둔다.

    왜 기본값을 "time"으로 두는가
    ----------------------------
    본 데이터는 8일치이며 일별 불량률이 5.0% ~ 54.4%로 크게 요동친다.
    랜덤 분할은 이 8일을 모두 섞어버리므로, 평가 대상 날짜의 데이터를
    이미 학습한 상태가 된다. 실제 현장에서는 성립할 수 없는 조건이다.

    동일 조건(Set_A, RandomForest)에서 실측한 결과:

        층화 랜덤 분할   ROC-AUC 0.9345
        시간 기준 분할   ROC-AUC 0.8417   <- 9.3%p 낮음

    즉 랜덤 분할은 성능을 약 9%p 부풀린다. 실운영 기준 성능은 후자다.

    Parameters
    ----------
    df : pd.DataFrame
        build_dataset()의 반환값.
    feature_set : {"A", "B"}, default "A"
    test_size : float, default 0.25
    seed : int, default 42
        strategy="random"일 때만 사용된다.
    hsb_only : bool, default False
        True면 hsb == '적용' 행(953건)만 사용하고 hsb 컬럼을 제외한다.
        "HSB를 걸었는데도 왜 불량이 났나"를 분석할 때 사용한다.
    strategy : {"time", "random"}, default "time"
    verbose : bool, default True

    Returns
    -------
    (x_train, x_test, y_train, y_test, preprocessor)
    """
    if strategy not in ("time", "random"):
        raise ValueError(f"strategy는 'time' 또는 'random'이어야 합니다: {strategy!r}")

    cat_cols, num_cols = get_feature_columns(feature_set)

    data = df.copy()
    if hsb_only:
        data = data[data["hsb"] == "적용"].reset_index(drop=True)
        cat_cols = [c for c in cat_cols if c != "hsb"]

    # 시간 기준 분할은 날짜 순서가 전제이므로 항상 정렬해둔다.
    # (build_dataset이 이미 정렬하지만, 사용자가 중간에 필터링했을 수 있다)
    data = data.sort_values(COL_DATE).reset_index(drop=True)

    x_all = data[cat_cols + num_cols]
    y_all = data[COL_TARGET]

    if strategy == "time":
        # 앞 (1 - test_size) 구간을 학습, 뒤 test_size 구간을 평가로 사용
        n_train = int(len(data) * (1 - test_size))
        x_train, x_test = x_all.iloc[:n_train], x_all.iloc[n_train:]
        y_train, y_test = y_all.iloc[:n_train], y_all.iloc[n_train:]
        cut_date = data[COL_DATE].iloc[n_train]
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x_all,
            y_all,
            test_size=test_size,
            stratify=y_all,      # 불량 비율을 train/test에 동일하게 유지
            random_state=seed,
        )
        cut_date = None

    prep = build_preprocessor(cat_cols, num_cols)

    if verbose:
        label = "시간 기준" if strategy == "time" else "층화 랜덤"
        msg = (
            f"[split] Set_{feature_set} ({FEATURE_SETS[feature_set]['desc']})"
            f"{' | HSB 적용분만' if hsb_only else ''}\n"
            f"        분할 방식 : {label}\n"
            f"        변수 {len(cat_cols) + len(num_cols)}개 "
            f"(범주 {len(cat_cols)} / 수치 {len(num_cols)})\n"
            f"        train {len(x_train)} (불량 {y_train.mean() * 100:.1f}%) | "
            f"test {len(x_test)} (불량 {y_test.mean() * 100:.1f}%)"
        )
        if cut_date is not None:
            msg += (
                f"\n        기준 시점 : {cut_date:%Y-%m-%d %H:%M} 이후가 평가 구간"
            )
        print(msg)

    return x_train, x_test, y_train, y_test, prep


# =============================================================================
# 동작 확인
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scale 불량 전처리 파이프라인 동작 확인"
    )
    parser.add_argument(
        "--data", default="data/SCALE불량.csv", help="원본 CSV 경로"
    )
    parser.add_argument(
        "--out", default=None, help="정제 결과 저장 경로 (.pkl)"
    )
    args = parser.parse_args()

    print("=" * 70)
    df_clean = build_dataset(args.data)
    print("=" * 70)

    for fs in FEATURE_SETS:
        split_dataset(df_clean, feature_set=fs)
        print("-" * 70)

    split_dataset(df_clean, feature_set="B", hsb_only=True)
    print("=" * 70)

    # --- engineer_one_row 와 add_domain_features 의 일치성 검증 ---
    # 두 경로가 어긋나면 대시보드 예측이 조용히 틀어진다.
    row = df_clean.iloc[0]
    src = {c: row[c] for c in [
        "fur_heat_temp", "fur_heat_time", "fur_soak_temp", "fur_soak_time",
        "fur_total_time", "pt_thick", "pt_width", "pt_length",
    ]}
    one = engineer_one_row(src)
    checks = ["heat_index", "soak_index", "soak_over_1173",
              "fur_wait_time", "sa_ratio", "aspect"]
    bad = [c for c in checks if abs(float(one[c]) - float(row[c])) > 1e-9]
    if bad:
        raise AssertionError(f"파생변수 불일치: {bad}")
    print(f"engineer_one_row 일치성 검증 통과 ({len(checks)}개 변수)")
    print("=" * 70)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_pickle(args.out)
        print(f"저장 완료 : {args.out}")
