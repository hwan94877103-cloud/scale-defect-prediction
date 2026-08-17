"""
대시보드용 아티팩트 빌드 스크립트
================================

Streamlit 앱이 실시간으로 무거운 연산을 하지 않도록, 다음을 미리 계산해
단일 pkl 파일로 저장한다.

    - Tab 1용 : Set_A + RandomForest  (실시간 경보)
    - Tab 2용 : Set_B + GradientBoosting (근본 원인 분석)
    - Permutation Importance (양쪽)
    - SHAP values (Set_B / GradientBoosting)
    - 입력 위젯 기본값 (학습셋 중앙값 / 최빈값)
    - 7개 모델 비교표, EDA용 원본 데이터

실행
----
    python build_artifacts.py --data data/SCALE불량.csv
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from models import run_comparison
from preprocessing import build_dataset, get_feature_columns

# Tab별 모델 역할 배정
#   Set_A에서는 RandomForest가 PR-AUC 1위이면서 회색지대도 가장 넓어
#   성능과 슬라이더 반응성을 모두 만족한다.
#   Set_B에서는 GradientBoosting이 FN 1건 / FP 0건으로 압도적이다.
TAB1_MODEL = "RandomForest"        # Set_A — 실시간 경보
TAB2_MODEL = "GradientBoosting"    # Set_B — 근본 원인 분석

DEFAULT_COST_RATIO = 5


def compute_importance(pipe, x_test, y_test, n_repeats=20, seed=42) -> pd.Series:
    """Permutation Importance를 계산한다.

    트리 모델의 기본 feature_importances_(불순도 기반)는 카디널리티가 높은
    변수를 과대평가하므로, 값을 섞었을 때의 성능 저하를 직접 측정하는
    Permutation 방식을 쓴다.
    """
    pi = permutation_importance(
        pipe, x_test, y_test,
        n_repeats=n_repeats, random_state=seed, scoring="f1", n_jobs=-1,
    )
    return pd.Series(pi.importances_mean, index=x_test.columns).sort_values(ascending=False)


def compute_shap(pipe, x_test):
    """SHAP values를 계산한다 (TreeExplainer).

    Returns
    -------
    (shap_values, x_encoded, feature_names) 또는 shap 미설치 시 (None, None, None)
    """
    try:
        import shap
    except ImportError:
        print("[warn] shap 미설치 → SHAP 탭은 비활성화됩니다.")
        return None, None, None

    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]

    x_enc = prep.transform(x_test)
    names = list(prep.get_feature_names_out())

    sv = shap.TreeExplainer(clf).shap_values(x_enc)
    if isinstance(sv, list):          # 구버전: [class0, class1]
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:  # 신버전: (n, feat, class)
        sv = sv[:, :, 1]

    return np.asarray(sv), np.asarray(x_enc), names


def make_input_defaults(df: pd.DataFrame, feature_set: str = "A") -> dict:
    """입력 위젯의 기본값과 범위를 만든다.

    오퍼레이터가 24개 변수를 전부 입력할 수는 없으므로, 화면에는 핵심 변수만
    노출하고 나머지는 여기서 만든 기본값(수치=중앙값, 범주=최빈값)으로 채운다.
    """
    cat_cols, num_cols = get_feature_columns(feature_set)

    defaults, ranges, options = {}, {}, {}

    for c in num_cols:
        s = df[c].dropna()
        defaults[c] = float(s.median())
        # 극단 이상치에 슬라이더가 끌려가지 않도록 1~99 퍼센타일로 자른다
        ranges[c] = (float(s.quantile(0.01)), float(s.quantile(0.99)))

    for c in cat_cols:
        defaults[c] = df[c].mode().iloc[0]
        options[c] = sorted(df[c].dropna().unique().tolist())

    return {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "defaults": defaults,
        "ranges": ranges,
        "options": options,
    }


def main(data_path: str, out_path: str, cost_ratio: int) -> None:
    print("=" * 70)
    df = build_dataset(data_path, verbose=True)
    print("=" * 70)

    # --- Set_A : 실시간 경보 ---
    print("\n[Set_A] 7개 모델 학습 중 ...")
    res_a = run_comparison(df, feature_set="A", cost_ratio=cost_ratio, verbose=False)
    pipe_a = res_a["models"][TAB1_MODEL]

    print(f"  · {TAB1_MODEL} Permutation Importance 계산 중 ...")
    imp_a = compute_importance(pipe_a, res_a["x_test"], res_a["y_test"])

    # --- Set_B : 근본 원인 분석 ---
    print("\n[Set_B] 7개 모델 학습 중 ...")
    res_b = run_comparison(df, feature_set="B", cost_ratio=cost_ratio, verbose=False)
    pipe_b = res_b["models"][TAB2_MODEL]

    print(f"  · {TAB2_MODEL} Permutation Importance 계산 중 ...")
    imp_b = compute_importance(pipe_b, res_b["x_test"], res_b["y_test"])

    print(f"  · {TAB2_MODEL} SHAP values 계산 중 ...")
    shap_values, x_enc, shap_names = compute_shap(pipe_b, res_b["x_test"])

    # --- 저장 ---
    artifact = {
        "built_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "cost_ratio_default": cost_ratio,

        # Tab 1
        "tab1": {
            "model_name": TAB1_MODEL,
            "pipeline": pipe_a,
            "x_test": res_a["x_test"],
            "y_test": res_a["y_test"],
            "prob_test": res_a["probs"][TAB1_MODEL],
            "summary": res_a["summary"],
            "importance": imp_a,
            "input_spec": make_input_defaults(df, "A"),
        },

        # Tab 2
        "tab2": {
            "model_name": TAB2_MODEL,
            "pipeline": pipe_b,
            "x_test": res_b["x_test"],
            "y_test": res_b["y_test"],
            "prob_test": res_b["probs"][TAB2_MODEL],
            "summary": res_b["summary"],
            "importance": imp_b,
            "shap_values": shap_values,
            "shap_x": x_enc,
            "shap_names": shap_names,
        },

        # EDA용 원본 (경량 컬럼만)
        "eda": df[[
            "scale", "target", "rolling_date",
            "fur_heat_temp", "fur_soak_temp", "rolling_temp",
            "fur_heat_time", "fur_soak_time", "fur_total_time",
            "hsb", "steel_kind", "spec_country", "fur_no", "work_group",
            "pt_thick", "pt_width", "pt_length", "descaling_count",
        ]].copy(),
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(artifact, f)

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\n>> 저장 완료: {out}  ({size_mb:.1f} MB)")
    print(f"   Tab1: Set_A + {TAB1_MODEL}")
    print(f"   Tab2: Set_B + {TAB2_MODEL}")
    print(f"\n실행: streamlit run app.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="대시보드 아티팩트 빌드")
    p.add_argument("--data", default="data/SCALE불량.csv")
    p.add_argument("--out", default="artifacts/dashboard.pkl")
    p.add_argument("--k", type=int, default=DEFAULT_COST_RATIO)
    args = p.parse_args()
    main(args.data, args.out, args.k)
