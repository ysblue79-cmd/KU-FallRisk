import os
import joblib
import pandas as pd
import altair as alt
import streamlit as st

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------
# 페이지 기본 설정 (항상 맨 위에서 한 번만 호출)
# -------------------------------------------------
st.set_page_config(
    page_title="낙상위험예측 (KU Medicine Digital Literacy)",
    page_icon="🧸",
    layout="wide",
)

# -------------------------------------------------
# 1) 메인 패널 펼침 버튼 제거 + keyboard_* 아이콘 숨기기
# -------------------------------------------------
custom_css = """
<style>
/* 사이드바 접기/펼치기 버튼(왼쪽 상단 화살표) 완전 숨기기 */
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
    visibility: hidden !important;
}

/* Material 아이콘 텍스트 (keyboard_double_arrow_*) 숨기기 */
[data-testid="stIconMaterial"] {
    font-size: 0 !important;
    visibility: hidden !important;
    color: transparent !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -------------------------------------------------
# UI 스타일 패키지 (최종 디테일 튜닝 + 구분선 수정 포함)
# -------------------------------------------------
ui_polish_css = """
<style>
/* --------------------------------------------- */
/* 0) Streamlit 기본 구분선 제거 (제목 아래 선 제거) */
/* --------------------------------------------- */
.block-container h1::before,
.block-container h2::before,
.block-container h3::before {
    border: none !important;
}

/* 1) 전체 기본 폰트/색 */
html, body, [class*="css"]  {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Malgun Gothic', sans-serif;
    color: #31333F;
}

/* 2) 메인 페이지 타이틀(H1) – 중앙 정렬 + 여백 */
h1 {
    text-align: center !important;
    margin-top: 0.3rem !important;
    margin-bottom: 0.2rem !important;
    font-weight: 700 !important;
}

/* 3) 섹션 타이틀(H2) */
h2 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    margin-top: 1.6rem !important;
    margin-bottom: 0.6rem !important;
}

/* 4) KPI 숫자 스타일 */
[data-testid="stMetricValue"] {
    font-size: 2.3rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.85rem !important;
    color: rgba(49, 51, 63, 0.7) !important;
}

/* 5) 단일 환자 결과 카드 스타일 */
.single-result-card {
    border-radius: 10px;
    padding: 10px 18px;
    font-size: 0.95rem;
}
.single-result-low    { background-color: #E6F9EC; border: 1px solid #82E0AA; }
.single-result-medium { background-color: #FEF3D4; border: 1px solid #F8C471; }
.single-result-high   { background-color: #FDE2E0; border: 1px solid #F5B7B1; }

/* 6) 위험군 칩(pill) */
.risk-chip {
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 600;
    display: inline-block;
    margin-right: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.risk-low    { background-color: #D5F5E3; }
.risk-medium { background-color: #F9E79F; }
.risk-high   { background-color: #FADBD8; }

/* 7) High Risk 테이블 제목 */
.highrisk-title {
    font-size: 1.0rem;
    font-weight: 600;
    margin-top: 1.8rem;
    margin-bottom: 0.4rem;
}
[data-testid="stTable"] {
    margin-top: 0.6rem;
}

/* 8) 사이드바 스타일 */
[data-testid="stSidebar"] {
    background-color: #F0F2F6;
    padding: 1.2rem 0.9rem 1.5rem 1.2rem;
}

/* 9) 스트림릿 기본 버튼/요소들 전체적으로 더 깔끔하게 */
.stButton>button {
    border-radius: 8px;
    padding: 0.45rem 1rem;
    font-weight: 600;
}
</style>
"""
st.markdown(ui_polish_css, unsafe_allow_html=True)

# -------------------------------------------------
# 3) Title & Logo (로고 왼쪽, 제목 가운데)
# -------------------------------------------------
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    if os.path.exists("hospital_logo.png"):
        st.image("hospital_logo.png", width=150)

with col2:
    st.markdown(
        """
        <h1 style="
            text-align: center;
            margin-top: 0.2rem;
            font-weight: 700;
            white-space: nowrap;
        ">
           AI 낙상 위험 예측 솔루션 (KU Medicine Digital Literacy)
        </h1>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.write("")  # 오른쪽 여백용

# 상단 여백 (구분선 대신)
st.markdown(
    """
    <div style="margin-top: 0.4rem; margin-bottom: 1.0rem;"></div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# 데이터 & 모델 로딩 (Cloud 자동 학습/로드 버전)
# -------------------------------------------------
DATA_PATH = "fall_data_simulated.csv"
MODEL_PATH = "rf_fall_model.pkl"


@st.cache_resource(show_spinner="📦 데이터를 불러오는 중입니다...")
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("❌ fall_data_simulated.csv 파일이 없습니다. 먼저 1_data_simulation.py 를 실행해 주세요.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource(show_spinner="🤖 낙상 위험 예측 모델을 준비하는 중입니다... (최초 1회만 학습)")
def load_or_train_model(df: pd.DataFrame):
    """
    1) rf_fall_model.pkl 있으면 먼저 불러보고
    2) 없거나 로딩 실패하면 df로 새로 학습 후 MODEL_PATH에 저장
    """
    # 1) 기존 모델 로드 시도
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            if isinstance(model, dict):
                model = model.get("model", model)
            return model
        except Exception:
            st.warning("⚠️ 기존 rf_fall_model.pkl 로딩 실패 → 데이터를 이용해 새로 학습합니다.")

    # 2) 자동 학습
    feature_cols_train = [
        c for c in df.columns
        if c not in ["registration_number", "risk_group", "fall_event"]
    ]

    if "fall_event" not in df.columns:
        st.error("❌ 'fall_event' 컬럼이 없어 자동 학습을 할 수 없습니다.")
        st.stop()

    X = df[feature_cols_train]
    y = df["fall_event"]

    # 범주형 / 수치형 분리
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in feature_cols_train if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    pipeline.fit(X, y)
    pipeline.feature_cols_ = feature_cols_train  # 학습에 사용한 feature 기억

    try:
        joblib.dump(pipeline, MODEL_PATH)
    except Exception:
        pass

    return pipeline


# 실제 데이터/모델 로딩
df = load_data()
model = load_or_train_model(df)

# -------------------------------------------------
# 모델 입력 feature 목록 (실제 학습 시 사용한 컬럼 순서와 맞추기)
# -------------------------------------------------
# 1순위: 모델이 기억하고 있는 feature_cols_
if hasattr(model, "feature_cols_"):
    feature_cols = list(model.feature_cols_)
else:
    # 백업용: 수동 정의
    feature_cols = [
        "dept",
        "ward",
        "admission_type",
        "age",
        "sex",
        "bmi",
        "nutrition_score",
        "fall_history",
        "surgery_history",
        "hospital_days",
        "sedative_use",
        "antipsychotic_use",
        "opioid_use",
        "antihypertensive_use",
        "diuretic_use",
        "mobility_level",
        "balance_impairment",
        "adl_score",
        "cognitive_impairment",
        "delirium",
        "altered_consciousness",
        "dizziness",
        "hypotension",
        "pain_score",
        "toileting_issue",
        "vision_impairment",
        "room_environment_risk",
        "bedbell_distance",
        "companion_presence",
        "morse_score",
        "braden_score",
        "Na",
        "Hb",
    ]

# -------------------------------------------------
# 예측용 안전 함수 (feature 자동 정렬/보정)
# -------------------------------------------------
def safe_predict_proba(model, X_input: pd.DataFrame) -> float:
    """단일 환자 / 수동 입력 모두에서
    - 모델이 학습한 feature 순서(model_features)에 맞춰 정렬
    - df에서 학습 당시 dtype으로 강제 캐스팅
    - 누락된 컬럼은 0으로 채움
    후에 안전하게 predict_proba 수행
    """
    # 1) DataFrame 보장
    if not isinstance(X_input, pd.DataFrame):
        X_input = pd.DataFrame(X_input)

    # 2) 모델이 기억하는 feature 목록 우선 사용
    if hasattr(model, "feature_cols_"):
        model_features = list(model.feature_cols_)
    else:
        model_features = list(feature_cols)

    # 3) 누락된 컬럼은 0으로 채워 넣기
    missing_cols = [c for c in model_features if c not in X_input.columns]
    for col in missing_cols:
        X_input[col] = 0

    # 4) 컬럼 순서를 모델 학습 순서에 맞추기
    X_input = X_input[model_features].copy()

    # 5) 학습 데이터(df)의 dtype과 동일하게 캐스팅  ⭐중요 수정 부분
    try:
        X_input = X_input.astype(df[model_features].dtypes.to_dict())
    except Exception:
        # 혹시 일부 컬럼에서 캐스팅이 안 되더라도 예측은 계속 시도
        pass

    # 6) 최종 예측
    return float(model.predict_proba(X_input)[0, 1])



# -------------------------------------------------
# 전체 환자에 대해 AI 기준 위험군 컬럼 추가
#   - model_prob : 모델이 예측한 낙상 확률
#   - model_risk_group : Low / Medium / High (0.3, 0.7 기준)
# -------------------------------------------------
def add_model_risk_columns(df_source: pd.DataFrame, model) -> pd.DataFrame:
    df_temp = df_source.copy()

    if hasattr(model, "feature_cols_"):
        model_features = list(model.feature_cols_)
    else:
        model_features = list(feature_cols)

    # 누락된 컬럼은 0으로 채우기
    missing_cols = [c for c in model_features if c not in df_temp.columns]
    for col in missing_cols:
        df_temp[col] = 0

    X_all = df_temp[model_features].copy()
    probs = model.predict_proba(X_all)[:, 1]

    df_temp["model_prob"] = probs
    df_temp["model_risk_group"] = pd.cut(
        df_temp["model_prob"],
        bins=[-0.01, 0.3, 0.7, 1.01],
        labels=["Low", "Medium", "High"],
    )
    return df_temp


# df에 AI 기준 위험군 컬럼 붙이기
df = add_model_risk_columns(df, model)

# -------------------------------------------------
# 위험군별 간호중재 내용 정의
# -------------------------------------------------
INTERVENTIONS_COMMON = [
    "입원 시 및 상태 변화 시 표준화된 낙상위험 사정 도구로 평가하고, EMR·카드에 낙상위험을 표시한다.",
    "침상 난간 올리기, 침상·휠체어 브레이크 고정, 바닥 정리, 조명 확보 등 낙상 예방을 위한 환경을 정돈한다.",
    "환자·보호자에게 호출벨 사용법, 일어나기 전 호출 요청, 미끄럼 주의 등 기본 낙상 예방 교육을 시행한다.",
]

INTERVENTIONS_BY_RISK = {
    "Low": [
        "침상 난간 및 콜벨 위치를 설명하고 사용 방법을 교육한다.",
        "침상 주변 환경을 정리하고 바닥 물기·선·이불·짐 등을 최소화한다.",
        "적절한 신발/슬리퍼 착용을 안내한다.",
        "혼자 일어나다가 넘어질 수 있음을 설명하고, 필요 시 호출하도록 교육한다.",
        "Morse/Braden 등 낙상 위험도 재평가를 정기적으로 시행한다.",
    ],
    "Medium": [
        "침상 난간 및 콜벨 위치를 재확인하고 사용법을 재교육한다.",
        "침상 주변 환경 정리 및 미끄럼 방지 슬리퍼 착용 안내를 시행한다.",
        "필요 시 보행 보조도구(워커, 지팡이 등)를 적용하고 사용법을 지도한다.",
        "야간·새벽 시간대 라운딩 시 우선 순위 대상에 포함한다.",
        "배뇨·배변 욕구 호소 시 가능하면 동행하고, 진정제·수면제·진통제 투약 후 보행 상태를 관찰한다.",
        "상태 변화(수술/투약 변경 등) 시 낙상위험도를 재평가한다.",
    ],
    "High": [
        "침상 난간 및 콜벨 위치를 재확인하고, 침대 높이를 최저로 유지하며 침대 바퀴 잠금을 확인한다.",
        "미끄럼 방지 양말/슬리퍼 착용 여부를 확인하고 필요 시 착용시킨다.",
        "침상 주변 선·의자·이불 등 걸려 넘어질 수 있는 물건을 제거한다.",
        "야간/새벽 시간대 낙상 고위험 환자 집중 라운딩(예: 1–2시간 간격)을 시행한다.",
        "배뇨/배변 욕구 또는 이뇨제 투약 후 화장실 이동 시 동행한다.",
        "시력·청력 저하 환자의 보조기(안경·보청기 등) 착용 여부와 보관 위치를 안내한다.",
        "기립성 저혈압 가능성이 있는 환자는 ‘침상 → 걸터앉기 → 다리 흔들기 → 천천히 일어나기’ 순서로 교육한다.",
        "진정제·수면제·항우울제·항정신병제·마약성 진통제 등 다약제 복용 시 필요 시 담당의와 용량/약제 조정을 논의한다.",
        "낙상 고위험 환자는 EMR/침상 카드에 ‘낙상 고위험’으로 표시하고, 팀(의사·간호사·물리치료사 등)과 공유한다.",
        "혼자 일어나지 말고 콜벨을 먼저 누르도록 환자·보호자에게 반복 교육하고, 교육 내용을 EMR에 기록한다.",
    ],
}

# =================================================
# 사이드바: 필터 옵션 + 단일 환자 예측
# =================================================
st.sidebar.header("🔎 필터 옵션")

# 병동/진료과 선택
ward_options = ["전체"] + sorted(df["ward"].dropna().astype(str).unique().tolist())
dept_options = ["전체"] + sorted(df["dept"].dropna().astype(str).unique().tolist())

ward_selected = st.sidebar.selectbox("병동 선택", ward_options)
dept_selected = st.sidebar.selectbox("진료과 선택", dept_options)

# 병동/진료과 필터 적용
filtered_df = df.copy()
if ward_selected != "전체":
    filtered_df = filtered_df[filtered_df["ward"].astype(str) == ward_selected]
if dept_selected != "전체":
    filtered_df = filtered_df[filtered_df["dept"].astype(str) == dept_selected]

# -------------------------------------------------
# 단일 환자 예측
# -------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🩺 단일 환자 낙상위험 예측")

mode = st.sidebar.radio("입력 방식 선택", ["데이터 기반", "직접 입력"], index=0)
pred_result = None  # (risk, prob, patient_row 또는 None)

# 1) 데이터 기반 (등록번호 선택)
if mode == "데이터 기반":
    st.sidebar.caption("현재 선택된 병동/진료과에 해당하는 환자 중에서 선택합니다.")

    if "registration_number" not in filtered_df.columns:
        st.sidebar.error("registration_number 컬럼이 데이터에 없습니다.")
    else:
        id_list = (
            filtered_df["registration_number"]
            .dropna()
            .astype(int)
            .sort_values()
            .unique()
            .tolist()
        )
        if not id_list:
            st.sidebar.warning("선택된 조건에 해당하는 환자가 없습니다.")
        else:
            selected_id = st.sidebar.selectbox("환자 등록번호 선택", id_list)
            patient_row = filtered_df[
                filtered_df["registration_number"].astype(int) == selected_id
            ]

            if patient_row.empty:
                st.sidebar.warning("해당 환자 정보를 찾을 수 없습니다.")
            else:
                patient_row = patient_row.iloc[0]
                with st.sidebar.container():
                    st.markdown("**선택 환자 정보**")
                    st.info(
                        f"• 등록번호: {int(patient_row['registration_number'])}\n\n"
                        f"• 병동: {patient_row['ward']}\n\n"
                        f"• 진료과: {patient_row['dept']}"
                    )

                if st.sidebar.button("예측하기", key="predict_data"):
                    X = patient_row[feature_cols].to_frame().T
                    prob = safe_predict_proba(model, X)
                    if prob > 0.7:
                        risk = "High"
                    elif prob > 0.3:
                        risk = "Medium"
                    else:
                        risk = "Low"
                    pred_result = (risk, prob, patient_row)

# 2) 직접 입력
else:
    st.sidebar.caption("주요 항목을 직접 입력하여 시뮬레이션합니다.")

    # (1) 범주형: 진료과 / 병동 / 입원 경로
    dept_choices = sorted(df["dept"].dropna().unique().tolist())
    ward_choices = sorted(df["ward"].dropna().unique().tolist())
    adm_choices = sorted(df["admission_type"].dropna().unique().tolist())

    dept_manual = st.sidebar.selectbox("진료과 (dept)", dept_choices)
    ward_manual = st.sidebar.selectbox("병동 (ward)", ward_choices)
    adm_manual = st.sidebar.selectbox("입원 경로 (admission_type)", adm_choices)

    # (2) 숫자형 / 이진 변수 입력
    age = st.sidebar.number_input("나이", min_value=20, max_value=95, value=70, step=1)
    sex_label = st.sidebar.selectbox("성별", ["여성", "남성"])
    sex = 0 if sex_label == "여성" else 1

    bmi = st.sidebar.number_input("BMI", min_value=16.0, max_value=40.0, value=23.0, step=0.1)
    nutrition_score = st.sidebar.slider("영양 점수 (nutrition_score)", 0, 10, 5)
    fall_history = st.sidebar.selectbox("과거 낙상력 (fall_history)", [0, 1])
    surgery_history = st.sidebar.selectbox("최근 수술력 (surgery_history)", [0, 1])
    hospital_days = st.sidebar.number_input("입원 일수 (hospital_days)", 0, 60, 5)
    sedative_use = st.sidebar.selectbox("진정제 사용 (sedative_use)", [0, 1])
    antipsychotic_use = st.sidebar.selectbox("항정신병제 사용 (antipsychotic_use)", [0, 1])
    opioid_use = st.sidebar.selectbox("마약성 진통제 사용 (opioid_use)", [0, 1])
    antihypertensive_use = st.sidebar.selectbox("항고혈압제 사용 (antihypertensive_use)", [0, 1])
    diuretic_use = st.sidebar.selectbox("이뇨제 사용 (diuretic_use)", [0, 1])
    mobility_level = st.sidebar.slider("이동능력 (mobility_level)", 0, 3, 1)
    balance_impairment = st.sidebar.selectbox("평형장애 (balance_impairment)", [0, 1])
    adl_score = st.sidebar.slider("ADL 점수 (adl_score)", 0, 100, 80)
    cognitive_impairment = st.sidebar.selectbox("인지장애 (cognitive_impairment)", [0, 1])
    delirium = st.sidebar.selectbox("섬망 (delirium)", [0, 1])
    altered_consciousness = st.sidebar.selectbox("의식변화 (altered_consciousness)", [0, 1])
    dizziness = st.sidebar.selectbox("어지러움 (dizziness)", [0, 1])
    hypotension = st.sidebar.selectbox("저혈압 (hypotension)", [0, 1])
    pain_score = st.sidebar.slider("통증 점수 (pain_score)", 0, 10, 3)
    toileting_issue = st.sidebar.selectbox("배뇨/배변 문제 (toileting_issue)", [0, 1])
    vision_impairment = st.sidebar.selectbox("시력 저하 (vision_impairment)", [0, 1])
    room_environment_risk = st.sidebar.selectbox("병실 환경 위험요인 (room_environment_risk)", [0, 1])
    bedbell_distance = st.sidebar.slider("콜벨 거리 (bedbell_distance)", 0, 3, 1)
    companion_presence = st.sidebar.selectbox("상주 보호자 있음 (companion_presence)", [0, 1])
    morse_score = st.sidebar.slider("Morse 점수 (morse_score)", 0, 100, 45)
    braden_score = st.sidebar.slider("Braden 점수 (braden_score)", 6, 23, 15)
    Na_val = st.sidebar.number_input("혈청 Na (Na)", min_value=120.0, max_value=150.0, value=138.0, step=0.5)
    Hb_val = st.sidebar.number_input("혈색소 Hb (Hb)", min_value=8.0, max_value=18.0, value=13.0, step=0.2)

    if st.sidebar.button("예측하기", key="predict_manual"):
        manual_input = {
            "dept": dept_manual,
            "ward": ward_manual,
            "admission_type": adm_manual,
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "nutrition_score": nutrition_score,
            "fall_history": fall_history,
            "surgery_history": surgery_history,
            "hospital_days": hospital_days,
            "sedative_use": sedative_use,
            "antipsychotic_use": antipsychotic_use,
            "opioid_use": opioid_use,
            "antihypertensive_use": antihypertensive_use,
            "diuretic_use": diuretic_use,
            "mobility_level": mobility_level,
            "balance_impairment": balance_impairment,
            "adl_score": adl_score,
            "cognitive_impairment": cognitive_impairment,
            "delirium": delirium,
            "altered_consciousness": altered_consciousness,
            "dizziness": dizziness,
            "hypotension": hypotension,
            "pain_score": pain_score,
            "toileting_issue": toileting_issue,
            "vision_impairment": vision_impairment,
            "room_environment_risk": room_environment_risk,
            "bedbell_distance": bedbell_distance,
            "companion_presence": companion_presence,
            "morse_score": morse_score,
            "braden_score": braden_score,
            "Na": Na_val,
            "Hb": Hb_val,
        }

        X = pd.DataFrame([manual_input], columns=feature_cols)
        prob = safe_predict_proba(model, X)

        if prob > 0.7:
            risk = "High"
        elif prob > 0.3:
            risk = "Medium"
        else:
            risk = "Low"
        pred_result = (risk, prob, None)

# 메인 영역과 사이드바 사이 구분선
st.markdown(
    """
    <div style="
        margin-top: 8px;
        margin-bottom: 10px;
        border-bottom: 1px solid #eaeaea;
    "></div>
    """,
    unsafe_allow_html=True,
)

# =================================================
# 메인 화면: KPI, 단일 환자 결과, 분포, 중재, High Risk 테이블
# =================================================
# 1) KPI 카드
st.markdown("### 📊 병동/진료과별 위험 분포 요약")

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("총 환자 수", len(filtered_df))

if "fall_event" in filtered_df.columns:
    kpi2.metric("낙상과거력 환자수", int(filtered_df["fall_event"].sum()))
else:
    kpi2.metric("낙상과거력 환자수", "-")

# 👉 AI 기준 High Risk 수 (model_risk_group 기준)
kpi3.metric("High Risk 환자 수", int((filtered_df["model_risk_group"] == "High").sum()))

# -------------------------------------------------
# 2) 단일 환자 예측 결과
# -------------------------------------------------
st.markdown("### 🎯 단일 환자 예측 결과")

if pred_result is not None:
    risk, prob, patient_row = pred_result

    color_map = {
        "High": "#FADBD8",
        "Medium": "#F9E79F",
        "Low": "#D5F5E3",
    }

    st.markdown(
        f"""
        <div style="
            background-color:{color_map.get(risk, '#E8E8E8')};
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        ">
            예측 위험군: {risk}<br>
            낙상 확률: {prob * 100:.1f}%
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(int(prob * 100), 100))

    if patient_row is not None:
        st.markdown("**환자 정보**")
        st.markdown(f"• 수술력: {'있음' if patient_row['surgery_history'] == 1 else '없음'}")
        st.markdown(f"• 낙상 이력: {'있음' if patient_row['fall_history'] == 1 else '없음'}")
    else:
        st.caption("※ 직접 입력 모드: EMR 기반 세부 정보는 제공되지 않습니다.")

    st.markdown("**주요 위험 요인 설명:**")
    risk_explanations = []

    if patient_row is not None:
        if patient_row["braden_score"] < 18:
            risk_explanations.append("- **Braden 점수**가 낮음 (피부·기능 상태 저하 가능성)")
        if patient_row["cognitive_impairment"] == 1:
            risk_explanations.append("- **인지장애** 있음")
        if patient_row["hypotension"] == 1:
            risk_explanations.append("- **저혈압** 있음 (체위 변경 시 어지러움 가능)")
        if patient_row["antihypertensive_use"] == 1:
            risk_explanations.append("- **항고혈압제 사용** (기립성 저혈압 가능)")
        if patient_row["mobility_level"] > 1:
            risk_explanations.append("- **이동 능력 저하** (보행 불안정 가능)")
        if patient_row["sedative_use"] == 1 or patient_row["opioid_use"] == 1:
            risk_explanations.append("- **진정제/마약성 진통제 사용** (졸림/어지러움 가능)")
    else:
        st.caption("※ 직접 입력 모드에서는 세부 위험 요인 자동 분석이 제한됩니다.")

    if risk_explanations:
        for explanation in risk_explanations:
            st.markdown(explanation)
    else:
        st.caption("특이 위험 요인이 뚜렷하게 높지 않습니다. 임상 상황에 따라 추가 평가가 필요합니다.")

    with st.expander("임상 해석 및 권고", expanded=True):
        st.markdown("**권장 간호중재:**")
        for item in INTERVENTIONS_BY_RISK[risk]:
            st.markdown(f"- {item}")
        st.caption("※ 시뮬레이션 데이터 기반 데모입니다. 실제 임상 적용 전 병원 지침 및 의무기록을 반드시 확인하세요.")
else:
    st.info("왼쪽에서 환자를 선택하거나 정보를 입력하면 단일 환자 예측 결과가 여기에 표시됩니다.")

# -------------------------------------------------
# 3) 낙상위험군별 표준 중재 (한 번만, 깔끔하게)
# -------------------------------------------------
st.markdown("### 🩺 낙상위험군별 표준 중재")

with st.expander("📘 공통 기본 중재 (모든 위험군)", expanded=False):
    for i, item in enumerate(INTERVENTIONS_COMMON, 1):
        st.markdown(f"{i}. {item}")

current_risk = pred_result[0] if pred_result is not None else None
low_expanded = current_risk == "Low"
med_expanded = current_risk == "Medium"
high_expanded = current_risk == "High"

col_low, col_med, col_high = st.columns(3)
with col_low:
    with st.expander("🟢 Low (낮은 위험군)", expanded=low_expanded):
        for a in INTERVENTIONS_BY_RISK["Low"]:
            st.markdown(f"- {a}")
with col_med:
    with st.expander("🟡 Medium (중간 위험군)", expanded=med_expanded):
        for a in INTERVENTIONS_BY_RISK["Medium"]:
            st.markdown(f"- {a}")
with col_high:
    with st.expander("🔴 High (높은 위험군)", expanded=high_expanded):
        for a in INTERVENTIONS_BY_RISK["High"]:
            st.markdown(f"- {a}")

# -------------------------------------------------
# 4) 위험군 분포 차트 (AI 기준 model_risk_group 사용)
# -------------------------------------------------
st.markdown("### 위험군 분포")
st.caption("현재 선택된 필터(병동/진료과 등)에 해당하는 환자들의 AI 기반 위험군 분포입니다.")

if filtered_df.empty:
    st.info("선택된 조건에 해당하는 환자가 없습니다.")
else:
    risk_counts = filtered_df["model_risk_group"].value_counts()
    risk_ratios = filtered_df["model_risk_group"].value_counts(normalize=True)

    risk_summary = pd.DataFrame({
        "risk_group": risk_counts.index.astype(str),
        "count": risk_counts.values.astype(float),
        "ratio": risk_ratios.values.astype(float),
    })
    risk_summary["ratio_label"] = (risk_summary["ratio"] * 100).round(1).astype(str) + "%"

    bar_chart = (
        alt.Chart(risk_summary)
        .mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10)
        .encode(
            x=alt.X(
                "risk_group:N",
                title="위험군",
                sort=["Low", "Medium", "High"],
            ),
            y=alt.Y("count:Q", title="환자 수"),
            color=alt.Color(
                "risk_group:N",
                scale=alt.Scale(
                    domain=["Low", "Medium", "High"],
                    range=["#D5F5E3", "#F9E79F", "#FADBD8"],
                ),
                legend=None,
            ),
        )
        .properties(height=380)
    )

    text_labels = (
        alt.Chart(risk_summary)
        .mark_text(
            align="center",
            baseline="bottom",
            fontSize=14,
            fontWeight="normal",
            dy=-5,
            color="gray",
        )
        .encode(
            x=alt.X("risk_group:N", sort=["Low", "Medium", "High"]),
            y="count:Q",
            text="ratio_label",
        )
    )

    final_chart = bar_chart + text_labels
    st.altair_chart(final_chart, use_container_width=True)

# -------------------------------------------------
# 5) High Risk Table (필터 조건 반영, AI 기준 High)
# -------------------------------------------------
st.markdown("### ⚠️ High Risk 환자 목록")

high_df = filtered_df[filtered_df["model_risk_group"] == "High"].copy()

if high_df.empty:
    st.info("선택된 조건에 해당하는 High Risk 환자가 없습니다.")
else:
    show_cols = [
        "registration_number",
        "dept",
        "ward",
        "age",
        "sex",
        "bmi",
        "nutrition_score",
        "fall_history",
        "surgery_history",
        "hospital_days",
        "sedative_use",
        "antipsychotic_use",
        "opioid_use",
        "antihypertensive_use",
        "diuretic_use",
        "mobility_level",
        "balance_impairment",
        "adl_score",
        "morse_score",
        "braden_score",
        "Na",
        "Hb",
    ]
    show_cols = [c for c in show_cols if c in high_df.columns]

    st.markdown(f"**현재 조건에 해당하는 High Risk 환자 수: {len(high_df)}명**")
    st.dataframe(high_df[show_cols], use_container_width=True)

    if st.button("엑셀로 내보내기"):
        export_df = high_df[show_cols]
        export_df.to_excel("fall_risk_report_output.xlsx", index=False)
        st.success("✅ 선택된 병동/진료과 기준 High Risk 환자 목록을 fall_risk_report_output.xlsx 로 저장했습니다.")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; color:#999; font-size:13px;'>
    © Korea University Guro Hospital – Nursing Digital Literacy Competition 2025<br>
    Developed by Nursing Administration Department
    </p>
    """,
    unsafe_allow_html=True,
)
