# 후판공정 Scale 불량 예측 프로젝트

철강 후판(Heavy Plate) 공정에서 발생하는 **Scale(산화철) 불량**을 예측하고,
그 근본 원인을 현업 엔지니어가 납득할 수 있는 형태로 설명하는 프로젝트.

```
가열로(예열대 → 가열대 → 균열대) → 추출 → HSB(고압수 세척) → 압연 → Scale 판정
```

- **데이터**: 1,000개 슬래브 × 21개 공정 변수 (2023-01-03 ~ 01-10)
- **타깃**: `scale` (양품 690 / 불량 310)
- **최종 산출물**: 7개 분류 모델 비교 → XAI 해석 → Streamlit 대시보드

---

## 전처리 파이프라인 설계 의도

`preprocessing.py`는 단순한 데이터 정리 스크립트가 아니라,
**"이 모델을 현장에서 언제 쓸 것인가"** 라는 질문에 대한 답을 코드로 구현한 모듈이다.

### 1. 예측 시점 기준으로 변수를 분리했다 (핵심 설계)

`rolling_temp`, `descaling_count`, `rolling_method`는 **압연이 끝나야 확정되는 값**이다.
그런데 Scale 불량 판정 역시 압연 후에 이뤄진다.
즉 이 변수들을 예측에 사용하면 *결과를 보고 결과를 맞히는* 구조가 되어,
정확도는 높지만 현업이 조치할 시점은 이미 지나 있다.

그래서 Feature Set을 둘로 나눴다.

| | **Set_A** | **Set_B** |
|---|---|---|
| 용도 | 사전 예측 · 실시간 경보 | 사후 분석 · XAI |
| 포함 변수 | 가열로 추출 시점까지 (24개) | 전 공정 (27개) |
| 사용 시점 | 슬래브가 가열로를 나올 때 | 불량 판정 후 원인 규명 |
| 대시보드 | Tab 1 (RandomForest) | Tab 2 (GradientBoosting) |
| RF 기준 ROC-AUC | 0.852 | 0.993 |

Set_B가 더 정확하지만, `rolling_temp` 하나가 Permutation Importance를
독식하면서 나머지 변수 중요도가 사실상 0으로 눌린다.
반면 Set_A는 중요도가 8개 변수로 고르게 분산되고
(`pt_width`, `steel_kind`, `hsb`, `sa_ratio`, `fur_soak_temp` …),
**전부 가열로 추출 이전에 확정되는 값이라 실제 경보에 쓸 수 있다.**

> 대시보드의 실시간 경보 기능은 **Set_A 모델**로 구동하고,
> Set_B는 원인 분석 탭에서 XAI 해석용으로만 사용한다.

### 2. 분할은 시간 기준으로 한다

`rolling_date` 오름차순 정렬 후 앞 75%를 학습, 뒤 25%를 평가로 자른다
(`strategy="time"`, 기본값). 실제 조업은 "과거로 미래를 예측"하는 상황이기 때문이다.

본 데이터는 8일치이며 **일별 불량률이 5.0% ~ 54.4%로 크게 요동친다.**
랜덤 분할은 이 8일을 모두 섞으므로 평가 대상 날짜를 이미 학습한 상태가 된다.
동일 조건(Set_A, RandomForest) 실측 결과:

| 분할 방식 | ROC-AUC |
|---|---|
| 층화 랜덤 | 0.9345 |
| **시간 기준** | **0.8417** |

랜덤 분할은 성능을 **9.3%p 부풀린다.** 실운영 기준 성능은 후자다.

임계값 산출용 교차검증도 같이 `TimeSeriesSplit`으로 맞췄다.
분할만 바꾸고 `StratifiedKFold`를 두면 학습셋 내부에서 미래를 보고
임계값을 정하게 되어 모순이 생긴다.

> `TimeSeriesSplit(n_splits=5, test_size=75)`로 초기 학습 구간을 375건 확보한다.
> 기본 설정(초기 125건)은 첫 fold가 불량 6.4% 구간으로 학습해 불량 64% 구간을
> 예측하게 되어, OOF 확률이 붕괴하고 임계값이 0.01까지 내려간다(실측).

### 3. 표준화·결측 대체를 파이프라인 안에 가뒀다

전체 데이터에 `fit_transform`을 먼저 걸고 나서 train/test를 나누면
테스트셋의 평균·표준편차가 학습에 새어 들어간다.
`build_preprocessor()`가 반환하는 `ColumnTransformer`를 `Pipeline`에 넣으면
교차검증 fold마다 학습 데이터만으로 다시 fit되므로 이 문제가 구조적으로 차단된다.
`rolling_temp`의 결측 대체(`SimpleImputer`)도 같은 이유로 파이프라인 내부에 둔다.

### 4. 도메인 지식을 파생변수로 명시했다

모델이 스스로 찾아내기 어려운 야금학적 관계를 변수로 만들어 넣었다.

| 파생변수 | 근거 |
|---|---|
| `sa_ratio` = 표면적/체적 | 단위 질량당 산화 노출 정도. 스케일 분석에서 물리적으로 가장 타당 |
| `soak_over_1173` | 페이얄라이트(Fe₂SiO₄) 융점 1,173°C. 넘으면 액상이 지철 계면에 침투해 난제거성이 됨 |
| `heat_index`, `soak_index` | 산화량은 온도×시간의 곱에 지배됨 (아레니우스) |
| `fur_wait_time` = 총재로−가열−균열 | 예열대 체류 + 순수 생산지연을 분리 |
| `gap_min` | 직전 판과의 압연 간격. 재로시간이 늘어난 *원인*을 설명 |
| `grade`, `is_tm` | 66종 `spec_long`을 원핫하는 대신 의미 축으로 분해 |

### 5. 물리적으로 불가능한 값을 결측으로 되돌렸다

`rolling_temp == 0`인 행이 6건 있다. 상온 압연은 불가능하므로
이는 온도계 미측정·통신 두절을 0으로 코딩한 것이다.
그대로 두면 평균이 아래로 끌려가고 표준화가 왜곡되므로 `NaN`으로 변환한다.

### 6. 학습과 서빙이 같은 파생변수 함수를 쓴다

`engineer_one_row()`를 `preprocessing.py`에 두고 대시보드가 이를 호출한다.
계산식이 두 곳에 존재하면 한쪽만 수정했을 때 **에러 없이 조용히 틀린 예측**이
나온다(train/serve skew). `python preprocessing.py`가 두 경로의 일치를 검증한다.

### 7. HSB 완전분리는 제거하지 않고 그대로 남겼다

`hsb` 미적용 47건이 **100% 불량**이다. 통계적으로는 완전분리(perfect separation)라
경계 대상이지만, 실측 결과 이 47건은 전체 불량 310건의 **15.2%**에 불과해
나머지 263건은 다른 변수가 설명해야 한다.
실제 Permutation Importance에서도 `hsb`는 1위가 아닌 2위였다.

따라서 **"HSB는 스케일 제거의 최후 방어선"이라는 도메인 팩트를
모델 중요도로 드러내기 위해** 그대로 포함시켰다.
다만 `split_dataset(..., hsb_only=True)` 옵션으로
*"HSB를 걸었는데도 왜 불량이 났나"* 를 보는 953건 서브 분석도 가능하도록 했다.

### 알려진 한계

- **`fur_ex_temp`(추출대 온도)가 데이터에 없다.** 도메인상 스케일 불량의
  최종 방아쇠로 알려진 변수인데, 본 데이터셋에서는 `rolling_temp`가 그 역할을
  대신하고 있다. 확보 시 예측력 향상이 기대된다.
- **`descaling_count`에 역인과 가능성이 있다.** 불량률이 횟수에 대해 단조롭지 않다
  (5회 100%, 6회 13.6%, 7회 100%, 8회 49%, 9회 100%, 10회 21.1%).
  스케일이 안 떨어지는 것을 보고 오퍼레이터가 추가 분사한 결과라면
  원인이 아니라 *결과의 대리 지표*다. 현장 확인 필요.
- **`steel_kind`의 C/T 정의가 미확인이다.** 통상 C=탄소강, T=TMCP 지정재로
  쓰이나 `rolling_method`의 TMCP(160건)와 T(245건) 개수가 일치하지 않는다.

---

## 사용법

```bash
pip install -r requirements.txt

# 동작 확인
python preprocessing.py --data data/SCALE불량.csv
```

```python
from preprocessing import build_dataset, split_dataset
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

df = build_dataset("data/SCALE불량.csv")

# 실시간 경보용 모델
x_tr, x_te, y_tr, y_te, prep = split_dataset(df, feature_set="A")

model = Pipeline([
    ("prep", prep),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
])
model.fit(x_tr, y_tr)
```

---

## 대시보드

```bash
python build_artifacts.py --data data/SCALE불량.csv   # 모델·SHAP 사전 계산
streamlit run app.py
```

### Tab 1 — 실시간 가열로 경보

Set_A + RandomForest. 슬래브가 가열로를 빠져나오는 시점의 조건을 입력하면
불량 위험도를 예측한다. 경보가 뜨면 압연 온도를 낮추거나 디스케일링을
추가하는 등의 조치가 가능하다.

**비용비 k 슬라이더 (k=1~10)** — 확률 임계값(0.26 등)은 현장에서 의미를 갖기
어려운 숫자다. 대신 *"불량 1건을 놓치는 것이 헛경보 몇 건에 해당하는가"* 를
물으면 품질팀과 생산팀이 협의해 답할 수 있고, 임계값은 `k × FN + FP` 최소화로
자동 계산된다.

**회색지대 안내** — 예측 확률이 0.05~0.95 구간이면 모델이 확신하지 못하는
상태이므로 육안 검사 병행을 안내한다.

### Tab 2 — 근본 원인 분석 · XAI

Set_B + GradientBoosting. 7개 모델 비교표, Permutation Importance,
SHAP Summary, 그리고 아레니우스 식 기반 도메인 인사이트를 제공한다.

### 모델 역할 배정 근거

| 화면 | Feature Set | 모델 | PR-AUC | 회색지대 |
|---|---|---|---|---|
| 실시간 경보 | Set_A | RandomForest | 0.926 | 83.2% |
| 원인 분석 | Set_B | GradientBoosting | 0.998 | 0.8% |

Set_A에서는 RandomForest가 PR-AUC 1위이면서 회색지대도 가장 넓어
**성능과 슬라이더 반응성을 모두 만족한다.**
Set_B에서는 GradientBoosting이 FN 1건 / FP 0건으로 압도적이지만
회색지대가 0.8%라 임계값을 조정해도 판정이 거의 바뀌지 않는다.
따라서 슬라이더가 필요한 실시간 경보에는 부적합하다.

---

## 프로젝트 구조

```
.
├── data/
│   └── SCALE불량.csv
├── preprocessing.py        # 전처리 · Feature Set 설계
├── models.py               # 7개 모델 비교 · 비용비 임계값 · TimeSeriesSplit
├── build_artifacts.py      # 대시보드용 사전 계산
├── app.py                  # Streamlit 대시보드
├── scale_defect_analysis.ipynb   # 튜토리얼 노트북
├── artifacts/
│   └── dashboard.pkl
├── requirements.txt
└── README.md
```
