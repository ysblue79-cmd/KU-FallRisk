import numpy as np
import pandas as pd

# ===============================================================
#  1_data_simulation.py
#  Restarted clean version with registration numbers (1,2,3,...)
# ===============================================================

def simulate_fall_data(n: int = 30000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    # -------------------------
    # Registration number
    # -------------------------
    registration_number = np.arange(1, n + 1)

    # -------------------------
    # Demographics / Basics
    # -------------------------
    age = rng.integers(20, 95, size=n)
    sex = rng.choice(["M", "F"], size=n)
    bmi = rng.normal(23, 3, size=n).clip(16, 38)
    nutrition_score = rng.normal(7, 2, size=n).clip(1, 10)
    fall_history = rng.binomial(1, 0.25, size=n)

    # -------------------------
    # Disease / Clinical Info
    # -------------------------
    dept = rng.choice(
        [
            "OBGY", "PU", "OS", "PS", "PD", "PY", "CA", "CS", "NE",
            "OL", "OPH", "BES", "CRS", "HPS", "GES", "ID", "NS",
            "NU", "GE", "L1", "EC", "RH", "HOO", "HOE"
        ],
        size=n,
    )
    ward = rng.choice(
        [
            "50", "51", "52", "53", "60", "61", "62", "63", "65",
            "70", "71", "72", "73", "80", "81", "82", "83",
            "90", "91", "92", "93", "95", "100", "101"
        ],
        size=n,
    )
    admission_type = rng.choice(["ER", "OPD", "DSC"], size=n, p=[0.4, 0.45, 0.15])
    surgery_history = rng.binomial(1, 0.3, size=n)
    hospital_days = rng.integers(1, 31, size=n)

    # -------------------------
    # Medication
    # -------------------------
    sedative_use = rng.binomial(1, 0.25, size=n)
    antipsychotic_use = rng.binomial(1, 0.15, size=n)
    opioid_use = rng.binomial(1, 0.2, size=n)
    antihypertensive_use = rng.binomial(1, 0.3, size=n)
    diuretic_use = rng.binomial(1, 0.25, size=n)

    # -------------------------
    # Physical / Mobility
    # -------------------------
    mobility_level = rng.binomial(1, 0.25, size=n)
    balance_impairment = rng.binomial(1, 0.2, size=n)
    adl_score = rng.normal(85, 15, size=n).clip(0, 100)

    # -------------------------
    # Cognitive / Neurologic
    # -------------------------
    cognitive_impairment = rng.binomial(1, 0.2, size=n)
    delirium = rng.binomial(1, 0.1, size=n)
    altered_consciousness = rng.binomial(1, 0.05, size=n)

    # -------------------------
    # Physiologic / Symptoms
    # -------------------------
    dizziness = rng.binomial(1, 0.25, size=n)
    hypotension = rng.binomial(1, 0.15, size=n)
    pain_score = rng.normal(3, 2, size=n).clip(0, 10)
    toileting_issue = rng.binomial(1, 0.2, size=n)
    vision_impairment = rng.binomial(1, 0.15, size=n)

    # -------------------------
    # Environment / Nursing
    # -------------------------
    room_environment_risk = rng.binomial(1, 0.1, size=n)
    bedbell_distance = rng.binomial(1, 0.1, size=n)
    companion_presence = rng.binomial(1, 0.7, size=n)

    # -------------------------
    # Classic Scores & Labs
    # -------------------------
    morse_score = rng.integers(0, 100, size=n)
    braden_score = rng.integers(6, 23, size=n)
    Na = rng.normal(138, 3, size=n).clip(120, 150)
    Hb = rng.normal(13, 1.5, size=n).clip(8, 18)

    # ===============================================================
    #  Assign fall-risk categories (Low / Medium / High)
    # ===============================================================
    weights = np.array([0.50, 0.35, 0.25])
    prob = weights / weights.sum()  # normalize to 1.0
    risk_group = rng.choice(["Low", "Medium", "High"], size=n, p=prob)

    base_prob = np.where(
        risk_group == "Low", 0.02,
        np.where(risk_group == "Medium", 0.08, 0.18)
    )
    modifier = (
        0.05 * fall_history
        + 0.04 * mobility_level
        + 0.03 * cognitive_impairment
        + 0.03 * dizziness
        + 0.03 * toileting_issue
        + 0.02 * room_environment_risk
    )
    prob_fall = np.clip(base_prob + modifier, 0, 0.9)
    fall_event = rng.binomial(1, prob_fall)

    # ===============================================================
    #  Combine into DataFrame
    # ===============================================================
    df = pd.DataFrame({
        "registration_number": registration_number,
        "dept": dept,
        "ward": ward,
        "admission_type": admission_type,
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
        "Na": Na,
        "Hb": Hb,
        "risk_group": risk_group,
        "fall_event": fall_event,
    })

    # Round all numeric columns to 1 decimal place
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(1)

    return df


if __name__ == "__main__":
    df = simulate_fall_data()
    df.to_csv("fall_data_simulated.csv", index=False)
    print("✅ fall_data_simulated.csv 생성 완료.")
    print(f"총 환자 수: {len(df)}")
    print(df["risk_group"].value_counts(normalize=True).round(2))
    print(f"낙상 발생률: {df['fall_event'].mean():.3f}")
