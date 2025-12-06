import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

# ===============================================================
#  2_train_model.py
#  Train RandomForest on expanded fall-risk dataset
# ===============================================================

DATA_PATH = "fall_data_simulated.csv"
MODEL_PATH = "rf_fall_model.pkl"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"{DATA_PATH} 파일이 없습니다. 먼저 1_data_simulation.py를 실행하세요."
    )

# ---------------------------------------------------------------
#  Load dataset
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# target
target_col = "fall_event"

# categorical + numeric feature sets
categorical_cols = [
    "sex", "dept", "ward", "admission_type",
]
numeric_cols = [
    "age", "bmi", "nutrition_score", "fall_history",
    "surgery_history", "hospital_days",
    "sedative_use", "antipsychotic_use", "opioid_use",
    "antihypertensive_use", "diuretic_use",
    "mobility_level", "balance_impairment", "adl_score",
    "cognitive_impairment", "delirium", "altered_consciousness",
    "dizziness", "hypotension", "pain_score",
    "toileting_issue", "vision_impairment",
    "room_environment_risk", "bedbell_distance", "companion_presence",
    "morse_score", "braden_score", "Na", "Hb",
]

feature_cols = categorical_cols + numeric_cols
X = df[feature_cols]
y = df[target_col]

# ---------------------------------------------------------------
#  Split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------
#  Preprocessing
# ---------------------------------------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# ---------------------------------------------------------------
#  Model
# ---------------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", rf_model)])

# ---------------------------------------------------------------
#  Train
# ---------------------------------------------------------------
pipeline.fit(X_train, y_train)
print("✅ 모델 학습 완료")

# ---------------------------------------------------------------
#  Evaluate
# ---------------------------------------------------------------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=3))
print(f"ROC-AUC: {auc:.3f}")

# ---------------------------------------------------------------
#  Save
# ---------------------------------------------------------------
joblib.dump(pipeline, MODEL_PATH)
print(f"✅ 모델 저장 완료: {MODEL_PATH}")
