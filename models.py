"""
후판공정 Scale 불량 예측 — 7개 모델 비교 및 비용비 기반 임계값 산출 모듈
========================================================================

`preprocessing.py`가 만든 Feature Set(A/B)을 받아 7개 분류 모델을 일괄 학습하고,
현업 비용 구조를 반영한 임계값을 산출한다.

핵심 설계
---------
1. **모델 순위는 임계값과 무관한 지표로 매긴다.**
   대시보드에서 임계값을 슬라이더로 조정할 예정이므로, F1@0.5로 줄을 세우면
   의미가 없다. 0.5에서 F1이 가장 높은 모델이 0.3에서도 최고라는 보장이 없다.
   따라서 ROC-AUC / PR-AUC(Average Precision)로 "줄 세우는 능력"을 먼저 본다.

2. **임계값은 비용비 k로 정한다.**
   "Precision 0.80 이상 유지하며 Recall 최대화" 같은 규칙은 실패한다.
   실제로 GradientBoosting에서 이 규칙을 적용하자 임계값 0.003을 골라
   불량 1건을 더 잡으려고 헛경보 19건을 만들어냈다.

   대신 `k * FN + FP`를 최소화한다. 여기서 k는 "불량 1건을 놓치는 것이
   헛경보 몇 건에 해당하는가"이다. 0.26 같은 확률값은 현업에게 의미 없는
   숫자지만, k는 엔지니어가 실제로 답할 수 있는 질문이다.

3. **임계값은 OOF 예측으로 정한다.**
   테스트셋에서 임계값을 찾으면 그 테스트 성능은 낙관 편향된다.
   `cross_val_predict`로 학습셋의 out-of-fold 확률을 얻어 임계값을 정하고,
   테스트셋에는 그 임계값을 그대로 적용해 최종 확인만 한다.

   **이때 교차검증 방식은 데이터 분할 방식과 반드시 일치해야 한다.**
   분할을 시간 기준으로 바꿔놓고 OOF는 StratifiedKFold(랜덤)로 뽑으면,
   "평가는 시간 순서를 지켰는데 임계값은 미래를 보고 정했다"는 모순이 생긴다.
   따라서 strategy="time"이면 TimeSeriesSplit을, "random"이면
   StratifiedKFold를 사용한다. 비용비 k와 비용식(k x FN + FP)은 그대로다.

4. **회색지대(gray zone)를 함께 측정한다.**
   예측 확률이 0.05~0.95 구간에 있는 샘플 비율. 슬라이더를 움직였을 때
   판정이 실제로 바뀔 수 있는 여지를 뜻한다. 이 값이 낮으면 성능이 아무리
   좋아도 대시보드 슬라이더가 반응하지 않는다.
   (실측: GradientBoosting 0.8% vs RandomForest 70.8%)

사용 예시
--------
    >>> from preprocessing import build_dataset
    >>> from models import run_comparison
    >>> df = build_dataset("data/SCALE불량.csv")
    >>> result = run_comparison(df, feature_set="A", cost_ratio=5)
    >>> print(result["summary"])

Author  : (your name)
License : MIT
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing import (
    DEFAULT_SEED,
    FeatureSetName,
    SplitStrategy,
    split_dataset,
)

warnings.filterwarnings("ignore")

__all__ = [
    "build_models",
    "build_cv",
    "oof_predict",
    "find_cost_threshold",
    "gray_zone_ratio",
    "evaluate_model",
    "run_comparison",
    "MAIN_MODEL",
    "UX_MODEL",
    "DEFAULT_COST_RATIO",
    "TS_TEST_SIZE",
]


# =============================================================================
# 상수
# =============================================================================

# 대시보드 메인 모델 — 성능 최고, FN 최소화
MAIN_MODEL = "GradientBoosting"

# 대시보드 비교/UX 보완 모델 — 회색지대가 넓어 슬라이더 체감이 좋음
UX_MODEL = "RandomForest"

# 기본 비용비. 불량 1건 놓침 = 헛경보 5건.
# 스케일 불량이 고객사로 출하되면 클레임·반품·신뢰도 손실로 이어지는 반면,
# 헛경보는 재검사 공수만 발생하므로 비대칭이 크다.
DEFAULT_COST_RATIO = 5

# 회색지대 판정 구간
GRAY_LOW, GRAY_HIGH = 0.05, 0.95

N_SPLITS = 5

# TimeSeriesSplit의 fold당 검증 구간 크기.
# 작게 잡을수록 1회차 학습 구간이 커진다 (build_cv 주석 참조).
TS_TEST_SIZE = 75


# =============================================================================
# 1) 모델 정의
# =============================================================================

def build_models(seed: int = DEFAULT_SEED) -> dict[str, object]:
    """비교 대상 7개 분류 모델을 반환한다.

    class_weight는 걸지 않는다.
    불균형(69:31) 보정은 임계값 튜닝이 전담한다. 둘 다 걸면 이중 보정이 되어
    "임계값을 k에서 유도한다"는 해석이 흐려지고, 슬라이더와 모델 내부 가중치가
    같은 일을 두 번 하게 된다.

    Returns
    -------
    dict[str, estimator]
        모델명 → 미학습 추정기.
    """
    return {
        # 해석 기준선. 계수 부호로 각 변수의 방향(위험↑/위험↓)을 읽을 수 있다.
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=seed),

        # 현업 설명용. 학습된 트리에서 "균열대 1170°C 초과 & HSB 미적용 → 불량"
        # 같은 규칙을 그대로 뽑아 표준작업서에 넣을 수 있다.
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=10, random_state=seed
        ),

        # 배깅 대표. 회색지대가 넓어 대시보드 슬라이더 체감이 좋다.
        "RandomForest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),

        # 부스팅 기본형. 본 데이터에서 성능 최고.
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=seed
        ),

        # 부스팅 고성능. 학습 속도가 GB보다 5배 빠르다.
        "XGBoost": _build_xgb(seed),

        # 거리 기반. 표준화가 제대로 됐는지 검증하는 역할도 겸한다.
        # 스케일링이 빠지면 이 모델의 성능이 가장 먼저 무너진다.
        "KNN": KNeighborsClassifier(n_neighbors=11, weights="distance"),

        # 마진 기반 비선형.
        "SVM_RBF": SVC(
            kernel="rbf", C=3.0, gamma="scale", probability=True, random_state=seed
        ),
    }


def _build_xgb(seed: int):
    """XGBoost는 선택 의존성으로 둔다 (미설치 시 GradientBoosting으로 대체)."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("[warn] xgboost 미설치 → GradientBoosting(깊이 4)으로 대체합니다.")
        return GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4, random_state=seed
        )
    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )


# =============================================================================
# 2) 비용비 기반 임계값
# =============================================================================

def find_cost_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_ratio: float = DEFAULT_COST_RATIO,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """총 비용 `k * FN + FP`를 최소화하는 임계값을 찾는다.

    Parameters
    ----------
    y_true : array-like
        실제 라벨 (불량=1).
    y_prob : array-like
        불량 예측 확률.
    cost_ratio : float, default 5
        k = 불량 1건 놓침의 비용 / 헛경보 1건의 비용.
        k가 크면 임계값이 내려가 더 공격적으로 불량을 잡는다.
    grid : ndarray, optional
        탐색할 임계값 격자. 기본 0.01~0.99, 0.01 간격.

    Returns
    -------
    (threshold, total_cost) : tuple[float, float]

    Notes
    -----
    이 함수는 반드시 **학습셋의 OOF 확률**로 호출해야 한다.
    테스트셋 확률로 임계값을 정하면 그 테스트 성능이 낙관 편향된다.
    """
    if grid is None:
        grid = np.arange(0.01, 1.00, 0.01)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    best_thr, best_cost = 0.5, np.inf
    for thr in grid:
        pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        cost = cost_ratio * fn + fp
        if cost < best_cost:
            best_thr, best_cost = float(thr), float(cost)

    return best_thr, best_cost


def gray_zone_ratio(y_prob: np.ndarray) -> float:
    """예측 확률이 회색지대(0.05~0.95)에 있는 샘플 비율.

    대시보드 슬라이더가 실제로 반응할 여지를 뜻한다.
    이 값이 0에 가까우면 모델이 모든 샘플을 확신하고 있어, 임계값을 어떻게
    움직여도 판정이 바뀌지 않는다 → 슬라이더가 사실상 장식이 된다.
    """
    p = np.asarray(y_prob)
    return float(((p > GRAY_LOW) & (p < GRAY_HIGH)).mean())


def oof_predict(pipe, x_train, y_train, cv):
    """교차검증 out-of-fold 확률을 수집한다.

    sklearn의 cross_val_predict는 사용할 수 없다. TimeSeriesSplit은 학습셋
    앞부분이 어떤 fold에서도 검증에 쓰이지 않아 전체를 덮는 분할(partition)이
    아니기 때문이며, cross_val_predict는 partition만 허용한다.

        ValueError: cross_val_predict only works for partitions

    따라서 fold를 직접 돌며 **검증 구간에 대해서만** 확률을 모은다.
    반환되는 y와 확률은 검증에 한 번이라도 쓰인 행만 포함하므로 길이가
    x_train보다 짧을 수 있다(TimeSeriesSplit의 경우 앞부분 제외).

    Returns
    -------
    (y_oof, prob_oof) : (np.ndarray, np.ndarray)
    """
    y_arr = np.asarray(y_train)
    ys, ps = [], []

    for tr_idx, va_idx in cv.split(x_train, y_arr):
        m = clone(pipe)
        m.fit(x_train.iloc[tr_idx], y_arr[tr_idx])
        ps.append(m.predict_proba(x_train.iloc[va_idx])[:, 1])
        ys.append(y_arr[va_idx])

    return np.concatenate(ys), np.concatenate(ps)


def build_cv(strategy: SplitStrategy = "time", n_splits: int = N_SPLITS,
             seed: int = DEFAULT_SEED):
    """분할 방식에 맞는 교차검증 객체를 만든다.

    ★ 이 함수가 존재하는 이유

    데이터 분할(train/test)과 임계값 산출용 교차검증은 **같은 논리를 따라야
    한다.** 분할만 시간 기준으로 바꾸고 교차검증은 StratifiedKFold(랜덤)로
    두면, 학습셋 내부에서 미래 데이터를 보고 임계값을 정하게 된다.

    예를 들어 학습셋이 1/03~1/08일 때 StratifiedKFold는 이렇게 자른다.

        1회차: 1/03,04,05,07,08 로 학습 -> 1/06 예측   <- 1/07,08 을 이미 봄

    TimeSeriesSplit은 항상 과거로 미래를 예측하는 구조만 만든다.

        1회차: 1/03          -> 1/04 예측
        2회차: 1/03,04       -> 1/05 예측
        3회차: 1/03,04,05    -> 1/06 예측
        4회차: 1/03,04,05,06 -> 1/07 예측

    비용비 k와 비용식(k x FN + FP)은 어느 쪽이든 동일하다.
    바뀌는 것은 그 계산에 들어가는 확률값을 어떻게 뽑느냐뿐이다.

    ★ test_size를 반드시 지정해야 하는 이유

    TimeSeriesSplit의 기본 동작은 학습셋을 (n_splits+1) 등분해 첫 조각만으로
    시작한다. 750건 / n_splits=5 이면 1회차 학습이 125건뿐이다.

    본 데이터는 일별 불량률이 5.0% ~ 54.4%로 요동치므로 이 경우 다음이 벌어진다.

        1회차: 학습 125건(불량 6.4%, 1/03~1/04) -> 검증 125건(불량 64.0%, 1/04~1/05)

    거의 양품만 본 모델이 불량 64%인 구간을 예측하니 확률이 전부 0 근처로
    깔리고, 이 왜곡된 확률이 OOF 전체를 오염시켜 임계값이 0.01까지 붕괴한다.
    그 임계값을 정상 학습된 모델의 테스트 확률에 적용하면 전부 불량으로
    판정되어 FP가 179건까지 치솟는다(실측).

    test_size를 작게 고정하면 1회차 학습 구간이 커진다.
    학습 750건 기준 실측 비교:

        n_splits=5 (기본)      1회차 학습 125건  OOF-AUC 0.617  thr 0.01  cost 179
        n_splits=3, test=125   1회차 학습 375건  OOF-AUC 0.838  thr 0.27  cost 125
        n_splits=5, test=75    1회차 학습 375건  OOF-AUC 0.871  thr 0.40  cost 105  <- 채택

    초기 학습 구간을 충분히 확보하면서 검증 횟수도 5회로 유지된다.

    Parameters
    ----------
    strategy : {"time", "random"}, default "time"
    n_splits : int, default 5
    seed : int
        strategy="random"일 때만 사용된다.

    Notes
    -----
    TimeSeriesSplit은 학습셋 앞부분(초기 학습 구간)이 어떤 fold에서도 검증에
    쓰이지 않으므로, OOF 확률의 개수가 학습셋보다 적다.
    """
    if strategy == "time":
        return TimeSeriesSplit(n_splits=n_splits, test_size=TS_TEST_SIZE)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


# =============================================================================
# 3) 단일 모델 평가
# =============================================================================

def evaluate_model(
    name: str,
    estimator,
    preprocessor,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    cost_ratio: float = DEFAULT_COST_RATIO,
    n_splits: int = N_SPLITS,
    seed: int = DEFAULT_SEED,
    strategy: SplitStrategy = "time",
) -> dict:
    """한 모델을 학습·평가하고 지표 일체를 반환한다.

    절차
    ----
    1. Pipeline(전처리 + 모델) 구성 — 표준화 누수 차단
    2. cross_val_predict로 학습셋 OOF 확률 획득
       (strategy에 맞는 교차검증 방식 사용 — build_cv 참조)
    3. OOF 확률에서 비용 최소 임계값 산출
    4. 전체 학습셋으로 재학습 → 테스트셋 확률 예측
    5. 3에서 정한 임계값을 그대로 적용해 최종 성능 측정

    Returns
    -------
    dict
        지표 + 학습된 파이프라인 + 확률 배열.
    """
    pipe = Pipeline([("prep", preprocessor), ("clf", estimator)])
    cv = build_cv(strategy, n_splits, seed)

    # --- (2) OOF 확률 (검증 구간만) ---
    y_oof, prob_oof = oof_predict(pipe, x_train, y_train, cv)

    # --- (3) 임계값은 OOF에서 결정 ---
    thr, _ = find_cost_threshold(y_oof, prob_oof, cost_ratio)

    # --- (4) 재학습 후 테스트 예측 ---
    pipe.fit(x_train, y_train)
    prob_test = pipe.predict_proba(x_test)[:, 1]

    # --- (5) 확정 임계값 적용 ---
    pred = (prob_test >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    return {
        "model": name,
        # 임계값 무관 지표 — 모델 순위는 이걸로 매긴다
        "ROC_AUC": roc_auc_score(y_test, prob_test),
        "PR_AUC": average_precision_score(y_test, prob_test),
        "ROC_AUC_oof": roc_auc_score(y_oof, prob_oof),
        # 비용비 기반 임계값
        "threshold": thr,
        # 확정 임계값에서의 성능
        "Recall": recall_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred),
        "Accuracy": accuracy_score(y_test, pred),
        "FN": int(fn),          # 놓친 불량 → 고객 클레임
        "FP": int(fp),          # 헛경보 → 재검사 공수
        "cost": int(cost_ratio * fn + fp),
        # 대시보드 슬라이더 반응성
        "gray_zone": gray_zone_ratio(prob_test),
        # 산출물
        "_pipeline": pipe,
        "_prob_test": prob_test,
        "_prob_oof": prob_oof,
    }


# =============================================================================
# 4) 7개 모델 일괄 비교
# =============================================================================

def run_comparison(
    df: pd.DataFrame,
    feature_set: FeatureSetName = "A",
    cost_ratio: float = DEFAULT_COST_RATIO,
    hsb_only: bool = False,
    seed: int = DEFAULT_SEED,
    strategy: SplitStrategy = "time",
    verbose: bool = True,
) -> dict:
    """7개 모델을 일괄 학습·평가하고 비교표를 만든다.

    Parameters
    ----------
    df : pd.DataFrame
        preprocessing.build_dataset()의 반환값.
    feature_set : {"A", "B"}, default "A"
    cost_ratio : float, default 5
    hsb_only : bool, default False
    seed : int, default 42
    verbose : bool, default True

    Returns
    -------
    dict
        summary      : 비교표 DataFrame (PR-AUC 내림차순)
        models       : 모델명 → 학습된 Pipeline
        probs        : 모델명 → 테스트 확률
        best         : PR-AUC 최우수 모델명
        x_test, y_test, feature_set, cost_ratio
    """
    x_tr, x_te, y_tr, y_te, prep = split_dataset(
        df, feature_set=feature_set, hsb_only=hsb_only, seed=seed,
        strategy=strategy, verbose=verbose,
    )

    rows, fitted, probs = [], {}, {}

    for name, est in build_models(seed).items():
        if verbose:
            print(f"  · {name} 학습 중 ...", flush=True)
        res = evaluate_model(
            name, est, prep, x_tr, y_tr, x_te, y_te,
            cost_ratio=cost_ratio, seed=seed, strategy=strategy,
        )
        fitted[name] = res.pop("_pipeline")
        probs[name] = res.pop("_prob_test")
        res.pop("_prob_oof")
        rows.append(res)

    summary = (
        pd.DataFrame(rows)
        .sort_values("PR_AUC", ascending=False)
        .reset_index(drop=True)
    )

    if verbose:
        _print_summary(summary, feature_set, cost_ratio, strategy)

    return {
        "summary": summary,
        "models": fitted,
        "probs": probs,
        "best": summary.iloc[0]["model"],
        "x_test": x_te,
        "y_test": y_te,
        "feature_set": feature_set,
        "cost_ratio": cost_ratio,
        "strategy": strategy,
    }


def _print_summary(summary: pd.DataFrame, feature_set: str, cost_ratio: float,
                   strategy: str = "time") -> None:
    cols = ["model", "ROC_AUC", "PR_AUC", "threshold",
            "Recall", "Precision", "F1", "FN", "FP", "cost", "gray_zone"]
    label = "시간 기준 분할 + TimeSeriesSplit" if strategy == "time" else "층화 랜덤 분할 + StratifiedKFold"
    print("\n" + "=" * 92)
    print(f"[Set_{feature_set}] 7개 모델 비교 — {label}")
    print(f"임계값 규칙: {cost_ratio}×FN + FP 최소화 (학습셋 OOF 기준)")
    print("=" * 92)
    print(summary[cols].round(3).to_string(index=False))
    print("-" * 92)
    print("FN = 놓친 불량(고객 클레임) | FP = 헛경보(재검사 공수) | "
          "gray_zone = 슬라이더 반응 여지")


# =============================================================================
# 5) 임계값 민감도 (대시보드 슬라이더 시뮬레이션)
# =============================================================================

def sweep_cost_ratio(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_values: range | list = range(1, 21),
) -> pd.DataFrame:
    """비용비 k를 훑으며 임계값과 성능이 어떻게 변하는지 표로 만든다.

    대시보드의 k 슬라이더가 실제로 무엇을 바꾸는지 미리 확인하는 용도이며,
    Streamlit 앱에서 그대로 재사용한다.
    """
    rows = []
    for k in k_values:
        thr, _ = find_cost_threshold(y_true, y_prob, k)
        pred = (np.asarray(y_prob) >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append({
            "k": k,
            "threshold": thr,
            "Recall": recall_score(y_true, pred),
            "Precision": precision_score(y_true, pred, zero_division=0),
            "FN": int(fn),
            "FP": int(fp),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 6) 산출물 저장
# =============================================================================

def save_artifacts(result: dict, out_dir: str | Path = "artifacts") -> Path:
    """Streamlit 대시보드에서 쓸 산출물을 저장한다.

    메인 모델(GradientBoosting)과 UX 보완 모델(RandomForest) 두 개를 함께 담는다.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = result["feature_set"]
    payload = {
        "summary": result["summary"],
        "feature_set": fs,
        "cost_ratio": result["cost_ratio"],
        "x_test": result["x_test"],
        "y_test": result["y_test"],
        "main_model": result["models"].get(MAIN_MODEL),
        "ux_model": result["models"].get(UX_MODEL),
        "prob_main": result["probs"].get(MAIN_MODEL),
        "prob_ux": result["probs"].get(UX_MODEL),
    }

    path = out_dir / f"artifact_set_{fs}.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


# =============================================================================
# 동작 확인
# =============================================================================

if __name__ == "__main__":
    import argparse

    from preprocessing import build_dataset

    parser = argparse.ArgumentParser(description="7개 모델 비교 실행")
    parser.add_argument("--data", default="data/SCALE불량.csv")
    parser.add_argument("--k", type=int, default=DEFAULT_COST_RATIO,
                        help="비용비 (불량 1건 놓침 = 헛경보 k건)")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--strategy", default="time", choices=["time", "random"],
                        help="분할 방식 (기본 time)")
    args = parser.parse_args()

    df_all = build_dataset(args.data, verbose=False)

    for fs in ("A", "B"):
        res = run_comparison(df_all, feature_set=fs, cost_ratio=args.k,
                             strategy=args.strategy)
        saved = save_artifacts(res, args.out)
        print(f">> 저장: {saved}\n")

    # 대시보드 슬라이더가 무엇을 바꾸는지 미리 확인
    res_a = run_comparison(df_all, feature_set="A", cost_ratio=args.k,
                           strategy=args.strategy, verbose=False)
    print("=" * 92)
    print(f"[Set_A / {MAIN_MODEL}] 비용비 k 슬라이더 시뮬레이션")
    print("=" * 92)
    print(sweep_cost_ratio(res_a["y_test"], res_a["probs"][MAIN_MODEL]).round(3).to_string(index=False))
