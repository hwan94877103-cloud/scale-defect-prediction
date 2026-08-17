"""
후판공정 Scale 불량 예측 대시보드
=================================

Tab 1 | 실시간 가열로 경보 시스템   — Set_A + RandomForest + 비용비 k 슬라이더
Tab 2 | 근본 원인 분석 및 XAI       — Set_B + GradientBoosting + SHAP

사전 준비
--------
    python build_artifacts.py --data data/SCALE불량.csv

실행
----
    streamlit run app.py
"""

from __future__ import annotations

import pickle
import os
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from models import find_cost_threshold, sweep_cost_ratio
from preprocessing import engineer_one_row

# =============================================================================
# 기본 설정
# =============================================================================

st.set_page_config(
    page_title="Scale 불량 예측 대시보드",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACT_PATH = Path("artifacts/dashboard.pkl")

C_OK, C_NG, C_WARN = "#2a6496", "#c0391b", "#c8901c"

# 화면에 노출할 핵심 입력 변수.
# 24개를 전부 입력받을 수는 없으므로, 오퍼레이터가 실제로 조정하거나 확인하는
# 변수만 노출하고 나머지는 학습셋 중앙값/최빈값으로 채운다.
KEY_NUM = [
    ("fur_soak_temp", "균열대 온도", "°C",
     "압연 직전 단계. 높으면 스케일이 단단하게 여물어 HSB로도 안 떨어진다."),
    ("fur_heat_temp", "가열대 온도", "°C",
     "표면 산화가 시작되는 구간. 불량의 도화선."),
    ("fur_soak_time", "균열대 재로시간", "분", "내부 온도 균일화를 위한 체류 시간."),
    ("fur_total_time", "총 재로시간", "분", "생산 지연 시 길어지며 스케일이 두꺼워진다."),
    ("pt_thick", "두께", "mm", "얇으면 압입 스케일(rolled-in), 두꺼우면 재로시간 증가."),
    ("pt_width", "폭", "mm", "넓으면 HSB 노즐 끝단 압력 부족으로 에지부에 잔류."),
]

KEY_CAT = [
    ("hsb", "HSB 적용", "고압수 디스케일러. 스케일 제거의 최후 방어선."),
    ("steel_kind", "강 종류", "C=탄소강 / T=TMCP 지정재(추정)."),
    ("fur_no", "가열로 호기", "버너 상태·공기비에 따라 실제 산화량이 다르다."),
    ("work_group", "작업조", "장입 간격·추출 타이밍 등 운전 습관 차이."),
]

# 그래프 라벨용 한글명
LABEL_KO = {
    "fur_heat_temp": "가열대 온도", "fur_soak_temp": "균열대 온도",
    "fur_heat_time": "가열대 시간", "fur_soak_time": "균열대 재로시간",
    "fur_total_time": "총 재로시간", "fur_wait_time": "대기시간",
    "rolling_temp": "압연 온도", "descaling_count": "디스케일링 횟수",
    "rolling_method": "압연 방식", "hsb": "HSB 적용",
    "pt_thick": "두께", "pt_width": "폭", "pt_length": "길이",
    "sa_ratio": "비표면적", "aspect": "세장비",
    "steel_kind": "강 종류", "spec_country": "규격 국가", "grade": "강도 등급",
    "is_tm": "TMCP 여부", "fur_no": "가열로 호기", "fur_input_row": "장입 열",
    "work_group": "작업조", "work_hour": "압연 시각", "gap_min": "압연 간격",
    "heat_index": "가열 열량지수", "soak_index": "균열 열량지수",
    "soak_over_1173": "페이얄라이트 임계 초과",
}


def setup_font() -> None:
    """한글 폰트 설정.

    이름으로만 폰트를 찾으면(fontManager.ttflist 스캔), 배포 환경에서 폰트를
    설치한 직후 matplotlib이 아직 그 폰트를 인식하지 못하는 경우가 있다.
    이를 막기 위해 apt로 설치되는 나눔고딕의 실제 파일 경로를 알고 있다면
    이름 검색보다 먼저 직접 등록한다 (packages.txt에 fonts-nanum 필요).

    seaborn을 쓸 경우 set_style() 이후에 폰트를 지정해야 한다.
    순서가 뒤바뀌면 rcParams가 덮어씌워져 한글이 □로 깨진다.
    """
    # Streamlit Cloud(Debian) 에서 fonts-nanum 설치 시 생성되는 경로.
    # 존재하면 이름 검색과 무관하게 직접 등록해 인식 실패를 막는다.
    for p in [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    ]:
        if os.path.exists(p):
            fm.fontManager.addfont(p)

    installed = {f.name for f in fm.fontManager.ttflist}
    for cand in ["NanumGothic", "Malgun Gothic", "AppleGothic",
                 "Noto Sans CJK KR", "Noto Sans KR"]:
        if cand in installed:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [cand]
            break
    plt.rcParams["axes.unicode_minus"] = False


setup_font()


@st.cache_resource(show_spinner="모델 로딩 중 ...")
def load_artifact(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def ko(name: str) -> str:
    """컬럼명을 한글 라벨로. 매핑이 없으면 원본을 그대로 쓴다."""
    return LABEL_KO.get(name, name)


# =============================================================================
# 아티팩트 로드
# =============================================================================

if not ARTIFACT_PATH.exists():
    st.error(
        f"아티팩트를 찾을 수 없습니다: `{ARTIFACT_PATH}`\n\n"
        "먼저 아래 명령을 실행하세요.\n\n"
        "```bash\npython build_artifacts.py --data data/SCALE불량.csv\n```"
    )
    st.stop()

ART = load_artifact(ARTIFACT_PATH)
TAB1, TAB2, EDA = ART["tab1"], ART["tab2"], ART["eda"]


# =============================================================================
# 사이드바
# =============================================================================

with st.sidebar:
    st.markdown("### 🔥 Scale 불량 예측")
    st.caption("후판공정 스마트팩토리 대시보드")
    st.divider()

    st.markdown("**모델 구성**")
    st.markdown(
        f"""
| 화면 | Feature Set | 모델 |
|---|---|---|
| 실시간 경보 | Set_A | {TAB1['model_name']} |
| 원인 분석 | Set_B | {TAB2['model_name']} |
"""
    )

    st.caption(
        "Set_A는 가열로 추출 시점에 알 수 있는 변수만 사용합니다. "
        "압연 온도·디스케일링 횟수는 압연이 끝나야 확정되므로, "
        "실시간 경보에 쓰면 조치 시점이 이미 지나 있습니다."
    )

    st.divider()
    row_a = TAB1["summary"].set_index("model").loc[TAB1["model_name"]]
    st.markdown("**Set_A 모델 성능** (홀드아웃)")
    c1, c2 = st.columns(2)
    c1.metric("Recall", f"{row_a['Recall']:.1%}")
    c2.metric("Precision", f"{row_a['Precision']:.1%}")
    c1.metric("PR-AUC", f"{row_a['PR_AUC']:.3f}")
    c2.metric("회색지대", f"{row_a['gray_zone']:.1%}")

    st.divider()
    st.caption(f"아티팩트 생성: {ART['built_at']}")


# =============================================================================
# Tab 구성
# =============================================================================

tab1, tab2 = st.tabs(["🚨  실시간 가열로 경보", "🔬  근본 원인 분석 · XAI"])


# =============================================================================
# Tab 1 — 실시간 경보
# =============================================================================

with tab1:
    st.markdown("## 실시간 가열로 경보 시스템")
    st.caption(
        "슬래브가 가열로를 빠져나오는 시점의 조건을 입력하면 Scale 불량 위험도를 예측합니다. "
        "경보가 뜨면 압연 온도를 낮추거나 디스케일링을 추가하는 등의 조치가 가능합니다."
    )

    spec = TAB1["input_spec"]
    left, right = st.columns([1, 1.4], gap="large")

    # ---------------------------------------------------------------- 입력
    with left:
        st.markdown("### 슬래브 조건")

        inputs = dict(spec["defaults"])   # 노출하지 않는 변수는 기본값 유지

        with st.expander("설비 · 강종", expanded=True):
            for col, label, help_txt in KEY_CAT:
                opts = spec["options"][col]
                inputs[col] = st.selectbox(
                    label, opts,
                    index=opts.index(spec["defaults"][col]),
                    help=help_txt,
                    key=f"cat_{col}",
                )

        with st.expander("가열로 조건", expanded=True):
            for col, label, unit, help_txt in KEY_NUM[:4]:
                lo, hi = spec["ranges"][col]
                inputs[col] = st.slider(
                    f"{label} ({unit})",
                    min_value=float(round(lo)), max_value=float(round(hi)),
                    value=float(round(spec["defaults"][col])),
                    step=1.0, help=help_txt, key=f"num_{col}",
                )

        with st.expander("제품 치수", expanded=False):
            for col, label, unit, help_txt in KEY_NUM[4:]:
                lo, hi = spec["ranges"][col]
                inputs[col] = st.slider(
                    f"{label} ({unit})",
                    min_value=float(round(lo)), max_value=float(round(hi)),
                    value=float(round(spec["defaults"][col])),
                    step=1.0, help=help_txt, key=f"num_{col}",
                )

        # 파생변수는 preprocessing 모듈의 함수를 그대로 호출한다.
        # 여기서 계산식을 다시 쓰면 학습 경로와 서빙 경로가 어긋나
        # 에러 없이 조용히 틀린 예측이 나온다 (train/serve skew).
        inputs = engineer_one_row(inputs)

    # ---------------------------------------------------------------- 예측
    with right:
        st.markdown("### 경보 민감도 설정")

        k = st.slider(
            "비용비 k — 불량 1건을 놓치는 것이 헛경보 몇 건에 해당합니까?",
            min_value=1, max_value=10, value=int(ART["cost_ratio_default"]), step=1,
            help=(
                "확률 임계값(0.26 등)은 현장에서 의미를 갖기 어려운 숫자입니다. "
                "대신 이 질문에 답하면 임계값이 자동으로 계산됩니다. "
                "k를 올리면 더 공격적으로 불량을 잡습니다."
            ),
        )

        # 임계값은 홀드아웃 확률 분포에서 k × FN + FP 최소화로 산출
        thr, _ = find_cost_threshold(TAB1["y_test"], TAB1["prob_test"], k)

        x_one = pd.DataFrame([inputs])[TAB1["x_test"].columns]
        prob = float(TAB1["pipeline"].predict_proba(x_one)[0, 1])
        is_alert = prob >= thr

        st.divider()

        if is_alert:
            st.error(f"### 🚨 경보 — 불량 위험\n\n불량 확률 **{prob:.1%}** (임계값 {thr:.2f} 초과)")
        else:
            st.success(f"### ✅ 정상\n\n불량 확률 **{prob:.1%}** (임계값 {thr:.2f} 미만)")

        m1, m2, m3 = st.columns(3)
        m1.metric("불량 확률", f"{prob:.1%}")
        m2.metric("적용 임계값", f"{thr:.2f}")
        m3.metric("판정", "경보" if is_alert else "정상")

        # --- 확률 위치 시각화 ---
        fig, ax = plt.subplots(figsize=(7, 1.7))
        ax.axvspan(0, thr, color=C_OK, alpha=.13)
        ax.axvspan(thr, 1, color=C_NG, alpha=.13)
        ax.axvspan(0.05, 0.95, color=C_WARN, alpha=.10)
        ax.axvline(thr, color="black", ls="--", lw=1.6)
        ax.scatter([prob], [0.5], s=340, color=C_NG if is_alert else C_OK,
                   zorder=5, edgecolors="white", linewidths=2)
        ax.text(prob, 0.78, f"{prob:.1%}", ha="center", fontsize=11, fontweight="bold")
        ax.text(thr, 0.16, f" 임계값 {thr:.2f}", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_yticks([]); ax.set_xlabel("불량 예측 확률")
        ax.set_title("판정 위치", fontsize=10, fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

        # --- 회색지대 안내 ---
        gray = float(TAB1["summary"].set_index("model")
                     .loc[TAB1["model_name"], "gray_zone"])

        if 0.05 < prob < 0.95:
            st.warning(
                f"**회색지대(0.05 ~ 0.95) 구간입니다.** "
                f"모델이 이 슬래브를 확신하지 못하고 있습니다. "
                f"k 값을 조정하면 판정이 바뀔 수 있으므로, 자동 판정에만 의존하지 말고 "
                f"육안 검사를 병행하십시오."
            )
        else:
            st.info(
                f"모델이 확신하는 구간(확률 {prob:.1%})입니다. "
                f"k를 조정해도 판정이 바뀌지 않을 가능성이 높습니다."
            )

        st.caption(
            f"참고 · 이 모델의 테스트셋 회색지대 비율은 **{gray:.1%}** 입니다. "
            "이 값이 낮은 모델은 임계값을 조정해도 결과가 거의 변하지 않아 "
            "슬라이더가 사실상 작동하지 않습니다."
        )

    # ---------------------------------------------------------------- k 민감도
    st.divider()
    st.markdown("### 비용비 k가 바꾸는 것")

    sweep = sweep_cost_ratio(TAB1["y_test"], TAB1["prob_test"], range(1, 11))

    c1, c2 = st.columns([1.4, 1], gap="large")

    with c1:
        fig, ax1 = plt.subplots(figsize=(8, 3.6))
        ax1.plot(sweep["k"], sweep["Recall"], marker="o", color=C_NG, lw=2.2, label="Recall")
        ax1.plot(sweep["k"], sweep["Precision"], marker="s", color=C_OK, lw=2.2, label="Precision")
        ax1.axvline(k, color="black", ls=":", lw=1.8)
        ax1.text(k, ax1.get_ylim()[1], f" 현재 k={k}", fontsize=9, va="top")
        ax1.set_xlabel("비용비 k"); ax1.set_ylabel("성능")
        ax1.set_xticks(range(1, 11)); ax1.legend(loc="lower left", fontsize=9)
        ax1.grid(alpha=.3)

        ax2 = ax1.twinx()
        ax2.bar(sweep["k"], sweep["threshold"], alpha=.15, color="gray")
        ax2.set_ylabel("적용 임계값"); ax2.grid(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        cur = sweep[sweep["k"] == k].iloc[0]
        st.markdown(f"**k = {k} 일 때 (테스트 250건 기준)**")
        st.dataframe(
            pd.DataFrame({
                "지표": ["적용 임계값", "Recall", "Precision",
                        "놓친 불량 (FN)", "헛경보 (FP)"],
                "값": [f"{cur['threshold']:.2f}", f"{cur['Recall']:.1%}",
                      f"{cur['Precision']:.1%}", f"{int(cur['FN'])}건",
                      f"{int(cur['FP'])}건"],
            }),
            hide_index=True,
        )
        st.caption(
            "k를 올리면 임계값이 내려가 불량을 더 많이 잡지만 헛경보도 늘어납니다. "
            "특정 k 이상에서는 변화가 포화되므로, 그 지점을 확인해 운영값을 정하십시오."
        )

    # ---------------------------------------------------------------- 중요도
    st.divider()
    st.markdown("### 이 모델이 보는 변수")

    imp = TAB1["importance"].head(10)[::-1]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.barh([ko(i) for i in imp.index], imp.values,
            color=[C_NG if v > 0.02 else "#9bb0c1" for v in imp.values])
    ax.set_xlabel("Permutation Importance (F1 감소량)")
    ax.set_title(f"Set_A · {TAB1['model_name']} — 상위 10개 변수",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =============================================================================
# Tab 2 — 근본 원인 분석 · XAI
# =============================================================================

with tab2:
    st.markdown("## 스케일 불량 근본 원인 분석")
    st.caption(
        "압연 공정 변수까지 포함한 사후 분석 화면입니다. "
        "실시간 예측에는 쓸 수 없지만, 불량이 왜 발생했는지 설명하는 데는 가장 정확합니다."
    )

    row_b = TAB2["summary"].set_index("model").loc[TAB2["model_name"]]
    m = st.columns(5)
    m[0].metric("모델", TAB2["model_name"])
    m[1].metric("Recall", f"{row_b['Recall']:.1%}")
    m[2].metric("Precision", f"{row_b['Precision']:.1%}")
    m[3].metric("놓친 불량 (FN)", f"{int(row_b['FN'])}건")
    m[4].metric("PR-AUC", f"{row_b['PR_AUC']:.3f}")

    st.divider()

    # ---------------------------------------------------------------- 모델 비교
    st.markdown("### 7개 모델 비교")

    cols_show = ["model", "ROC_AUC", "PR_AUC", "threshold", "Recall",
                 "Precision", "F1", "FN", "FP", "gray_zone"]

    cmp_set = st.radio(
        "Feature Set", ["Set_B (사후 분석)", "Set_A (사전 예측)"],
        horizontal=True, label_visibility="collapsed",
    )
    tbl = (TAB2 if cmp_set.startswith("Set_B") else TAB1)["summary"][cols_show]

    st.dataframe(
        tbl.style.format({
            "ROC_AUC": "{:.3f}", "PR_AUC": "{:.3f}", "threshold": "{:.2f}",
            "Recall": "{:.1%}", "Precision": "{:.1%}", "F1": "{:.3f}",
            "gray_zone": "{:.1%}",
        }).background_gradient(subset=["PR_AUC"], cmap="Blues"),
        hide_index=True,
    )
    st.caption(
        "모델 순위는 임계값과 무관한 PR-AUC로 매깁니다. 대시보드에서 임계값을 "
        "조정할 예정이므로 F1@0.5로 줄을 세우면 의미가 없습니다."
    )

    st.divider()

    # ---------------------------------------------------------------- 중요도 + SHAP
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("### 변수 중요도")
        imp_b = TAB2["importance"].head(12)[::-1]
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        ax.barh([ko(i) for i in imp_b.index], imp_b.values,
                color=[C_NG if v > 0.02 else "#9bb0c1" for v in imp_b.values])
        ax.set_xlabel("Permutation Importance")
        ax.set_title(f"Set_B · {TAB2['model_name']}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "불순도 기반 중요도는 카디널리티가 높은 변수를 과대평가하므로, "
            "값을 섞었을 때의 성능 저하를 직접 측정하는 Permutation 방식을 사용했습니다."
        )

    with c2:
        st.markdown("### SHAP 요약")
        if TAB2.get("shap_values") is not None:
            try:
                import shap
                fig = plt.figure(figsize=(6.5, 4.6))
                shap.summary_plot(
                    TAB2["shap_values"], TAB2["shap_x"],
                    feature_names=TAB2["shap_names"],
                    max_display=12, show=False, plot_size=None,
                )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.info(f"SHAP 렌더링 실패: {e}")
        else:
            st.info("SHAP 값이 아티팩트에 없습니다. `pip install shap` 후 재빌드하세요.")

        st.caption(
            "가로축 오른쪽(양수) = 불량 위험 상승. 점 색깔은 그 변수의 값 크기입니다. "
            "**빨간 점이 오른쪽에 몰려 있으면** 그 변수가 클수록 위험하다는 뜻입니다."
        )

    st.divider()

    # ---------------------------------------------------------------- 아레니우스
    st.markdown("### 도메인 인사이트 — 고온·단시간이 저온·장시간보다 위험하다")

    c1, c2 = st.columns([1, 1.25], gap="large")

    with c1:
        st.markdown(
            """
분석 과정에서 도메인 가설 하나가 데이터로 **뒤집혔습니다.**

| 가설 | 결과 |
|---|---|
| 온도가 높을수록 불량 | **지지** |
| 시간이 길수록 불량 | **부분 기각** |

온도 3종은 모두 불량 쪽이 높습니다(압연 온도 **+78.5°C**).
그런데 시간 3종은 모두 **불량 쪽이 오히려 짧습니다**.

가설이 틀린 것은 아닙니다. 산화 속도는 아레니우스 식
"""
        )
        st.latex(r"k = A \cdot e^{-E_a / RT}")
        st.markdown(
            """
을 따르므로 **온도에 대해서는 지수적**, **시간에 대해서는 선형**에 가깝습니다.
같은 열량을 투입해도

- **고온 · 단시간** → 산화 격렬
- **저온 · 장시간** → 산화 완만

이 됩니다.

> **현장 시사점**
> 재로시간을 줄이는 것보다 **온도 상한을 관리하는 쪽**이 개선 효과가 큽니다.
> 특히 페이얄라이트(Fe₂SiO₄) 융점 **1,173°C** 부근에서 액상 스케일이
> 지철 계면에 침투해 난제거성이 되므로, 이 온도를 운전 상한의
> 기준점으로 검토할 가치가 있습니다.
"""
        )

    with c2:
        # --- 필터 ---
        f1, f2 = st.columns(2)
        sel_kind = f1.multiselect(
            "강 종류", sorted(EDA["steel_kind"].unique()),
            default=sorted(EDA["steel_kind"].unique()),
        )
        sel_fur = f2.multiselect(
            "가열로 호기", sorted(EDA["fur_no"].unique()),
            default=sorted(EDA["fur_no"].unique()),
        )

        d = EDA[EDA["steel_kind"].isin(sel_kind) & EDA["fur_no"].isin(sel_fur)]

        if len(d) < 10:
            st.warning("필터 조건에 해당하는 데이터가 너무 적습니다.")
        else:
            fig, ax = plt.subplots(figsize=(7, 5.2))
            for t, lab, c in [(0, "양품", C_OK), (1, "불량", C_NG)]:
                s = d[d.target == t]
                ax.scatter(s["fur_soak_temp"], s["fur_soak_time"],
                           s=20, alpha=.5, c=c, label=lab, edgecolors="none")
            ax.axvline(1173, color="darkorange", ls="--", lw=2)
            ax.text(1173.5, ax.get_ylim()[1] * .97, " 페이얄라이트 융점 1,173°C",
                    color="darkorange", fontsize=9, fontweight="bold", va="top")
            ax.set_xlabel("균열대 온도 (°C)")
            ax.set_ylabel("균열대 재로시간 (분)")
            ax.set_title(f"고온·단시간 vs 저온·장시간  (n={len(d)}, "
                         f"불량률 {d.target.mean():.1%})",
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=9); ax.grid(alpha=.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "왼쪽 위(저온·장시간)에는 양품이, 오른쪽 아래(고온·단시간)로 갈수록 "
                "불량이 몰립니다."
            )

    st.divider()

    # ---------------------------------------------------------------- HSB
    st.markdown("### HSB — 스케일 제거의 최후 방어선")

    c1, c2, c3 = st.columns([1, 1, 1.1], gap="large")

    hsb_tab = EDA.groupby("hsb")["target"].agg(["size", "mean"])
    n_nohsb = int(((EDA.hsb == "미적용") & (EDA.target == 1)).sum())
    n_other = int(EDA.target.sum() - n_nohsb)

    with c1:
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        ax.bar(hsb_tab.index, hsb_tab["mean"] * 100,
               color=[C_NG, C_OK], width=.5)
        for i, (n, mv) in enumerate(zip(hsb_tab["size"], hsb_tab["mean"])):
            ax.text(i, mv * 100 + 2, f"{mv*100:.1f}%\n(n={n})",
                    ha="center", fontsize=9)
        ax.set_ylim(0, 120); ax.set_ylabel("불량률 (%)")
        ax.set_title("HSB 적용 여부별 불량률", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        ax.pie([n_nohsb, n_other],
               labels=[f"HSB 미적용\n{n_nohsb}건", f"기타 원인\n{n_other}건"],
               colors=[C_NG, "#e0b089"], autopct="%1.1f%%", startangle=90,
               wedgeprops=dict(width=.45, edgecolor="w"), textprops={"fontsize": 9})
        ax.set_title("불량 구성", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c3:
        st.markdown(
            f"""
HSB 미적용 **{int(hsb_tab.loc['미적용', 'size'])}건**은 예외 없이
**100% 불량**입니다. "HSB는 스케일 제거의 유일한 물리적 방어선"이라는
도메인 지식이 데이터로 완벽히 재현됩니다.

다만 이 건수는 전체 불량의 **{n_nohsb / (n_nohsb + n_other):.1%}** 에
불과합니다. 나머지 **{n_other}건**은 HSB를 정상 적용했는데도 불량이
났습니다.

> **조치 우선순위**
> 1. HSB 미적용 사유 조사 — 설비 트러블인지, 특정 강종 회피인지
>    확인이 필요합니다. 이건 모델링이 아니라 **운전 규칙 문제**입니다.
> 2. 광폭재 HSB 노즐 점검 — 에지부 충돌압력 부족 여부 실측
"""
        )

    st.divider()

    # ---------------------------------------------------------------- 한계
    with st.expander("분석의 한계 — 해석 시 주의사항", expanded=False):
        st.markdown(
            """
- **`fur_ex_temp`(추출대 온도)가 데이터에 없습니다.** 도메인상 스케일 불량의
  최종 방아쇠로 알려진 변수인데, 본 데이터셋에서는 `rolling_temp`가 그 역할을
  대신하고 있습니다. 확보 시 Set_A 성능 향상이 기대됩니다.

- **`descaling_count`에 역인과 가능성이 있습니다.** 불량률이 횟수에 대해
  단조롭지 않습니다(5회 100%, 6회 13.6%, 7회 100%, 8회 49%, 9회 100%, 10회 21.1%).
  스케일이 안 떨어지는 것을 보고 오퍼레이터가 추가 분사한 결과라면, 원인이 아니라
  **결과의 대리 지표**입니다. 현장 확인이 필요합니다.

- **불량률 31%는 실조업 수치가 아닙니다.** 학습용으로 오버샘플링된 것으로
  보이므로 절대값이 아닌 **상대 경향**만 해석해야 합니다.

- **데이터 기간이 8일로 짧습니다.** 계절 변동이나 설비 정비 주기 효과를
  담지 못합니다.

- **`steel_kind`의 C/T 정의가 미확인입니다.** 통상 C=탄소강, T=TMCP 지정재로
  쓰이나 `rolling_method`의 TMCP(160건)와 T(245건) 개수가 일치하지 않습니다.
"""
        )
