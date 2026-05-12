"""
Cardiovascular Disease Detection Tool
Streamlit GUI that wraps the trained multi-class CVD classifier
(Healthy / Coronary Artery Disease / Heart Failure).

Loads the four artefacts persisted by `final_code.ipynb`:
  Model/final_model.pkl       - full sklearn Pipeline (preprocessor + SVC)
  Model/scaler.pkl            - fitted StandardScaler (continuous block)
  Model/feature_columns.pkl   - exact column order the pipeline expects
  Model/label_mapping.pkl     - {class_name -> integer code}
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main > div { padding-top: 1rem; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }

    .app-title {
        font-size: 1.51rem; font-weight: 200; color: #b91c1c;
        margin-bottom: 0.1rem;
    }
    .app-sub {
        color: #475569; font-size: 1.0rem; margin-bottom: 1.2rem;
    }

    .section-head {
        font-size: 1.05rem; font-weight: 600; color: #0f172a;
        border-left: 4px solid #b91c1c; padding-left: 0.6rem;
        margin: 0.6rem 0 0.4rem 0;
    }

    .pred-card {
        padding: 1.2rem 1.4rem; border-radius: 14px;
        color: white; margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    }
    .pred-card h2 { margin: 0; font-size: 1.6rem; }
    .pred-card p { margin: 0.3rem 0 0 0; font-size: 0.95rem; opacity: 0.92; }

    .pred-healthy { background: linear-gradient(135deg, #16a34a, #15803d); }
    .pred-cad     { background: linear-gradient(135deg, #ea580c, #c2410c); }
    .pred-hf      { background: linear-gradient(135deg, #dc2626, #991b1b); }

    .metric-box {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 0.7rem 0.9rem;
    }
    .metric-label { color: #64748b; font-size: 0.78rem; }
    .metric-value { color: #0f172a; font-size: 1.15rem; font-weight: 600; }

    .footnote { color:#64748b; font-size:0.8rem; margin-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Artefact loading
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent / "Model"


@st.cache_resource(show_spinner="Loading trained model...")
def load_artefacts():
    model = joblib.load(MODEL_DIR / "final_model.pkl")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")
    label_mapping = joblib.load(MODEL_DIR / "label_mapping.pkl")
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    return model, feature_columns, label_mapping, inv_label_mapping


model, FEATURE_COLUMNS, LABEL_MAPPING, INV_LABEL_MAPPING = load_artefacts()
CLASS_NAMES = [INV_LABEL_MAPPING[i] for i in range(len(INV_LABEL_MAPPING))]

# ---------------------------------------------------------------------------
# Categorical option maps (must match the encoding used during training)
# ---------------------------------------------------------------------------
GENDER_OPTS = {"Female": 0, "Male": 1}
YESNO_OPTS = {"No": 0, "Yes": 1}
CHOL_OPTS = {
    "Desirable (<200 mg/dL)": 1,
    "Borderline (200-239 mg/dL)": 2,
    "High (>=240 mg/dL)": 3,
}
GLUC_OPTS = {
    "Normal (<100 mg/dL)": 1,
    "Prediabetes (100-125 mg/dL)": 2,
    "Diabetes (>=126 mg/dL)": 3,
}
CPT_OPTS = {
    "Typical Angina": 1,
    "Atypical Angina": 2,
    "Non-Anginal Pain": 3,
    "Asymptomatic": 4,
}
ECG_OPTS = {
    "Normal": 0,
    "ST-T Abnormality": 1,
    "Left Ventricular Hypertrophy": 2,
}
SLOPE_OPTS = {
    "Upsloping": 1,
    "Flat": 2,
    "Downsloping": 3,
}

PRED_CARD_CLASS = {
    "Healthy": "pred-healthy",
    "Coronary Artery Disease": "pred-cad",
    "Heart Failure": "pred-hf",
}

PRED_BLURB = {
    "Healthy":
        "No cardiovascular disease pattern detected in the supplied measurements.",
    "Coronary Artery Disease":
        "Pattern consistent with atherosclerotic narrowing of coronary arteries. "
        "Recommend follow-up with a cardiologist for clinical confirmation.",
    "Heart Failure":
        "Pattern consistent with reduced pumping capacity of the heart. "
        "Recommend prompt clinical evaluation.",
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="app-title"></div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">Multi-class screening of <b>Healthy</b>, '
    '<b>Coronary Artery Disease</b> and <b>Heart Failure</b> from 16 clinical '
    'inputs.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar - presets & info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Quick Presets")
    preset = st.radio(
        "Pre-fill the form with a clinical profile:",
        ["None (manual entry)", "Healthy profile", "CAD profile", "HF profile"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This tool wraps a tuned **SVM (RBF)** pipeline trained on a "
        "45,000-row synthetic cardiovascular cohort. Three engineered "
        "features (`bmi`, `pulse_pressure`, `chronotropic_response_index`) "
        "are computed automatically before scoring."
    )
    st.caption("Trained pipeline: final_model.pkl")

PRESET_VALUES = {
    "Healthy profile": dict(
        age=45, gender="Male", height_cm=170, weight_kg=70,
        systolic_bp=118, diastolic_bp=76,
        cholesterol="Desirable (<200 mg/dL)",
        glucose="Normal (<100 mg/dL)",
        smoking="No", alcohol="No",
        cpt="Atypical Angina", ecg="Normal",
        max_hr=160, ex_angina="No", oldpeak=0.2, slope="Upsloping",
    ),
    "CAD profile": dict(
        age=62, gender="Male", height_cm=168, weight_kg=85,
        systolic_bp=150, diastolic_bp=92,
        cholesterol="High (>=240 mg/dL)",
        glucose="Prediabetes (100-125 mg/dL)",
        smoking="Yes", alcohol="Yes",
        cpt="Asymptomatic", ecg="ST-T Abnormality",
        max_hr=125, ex_angina="Yes", oldpeak=2.4, slope="Flat",
    ),
    "HF profile": dict(
        age=70, gender="Female", height_cm=164, weight_kg=85,
        systolic_bp=138, diastolic_bp=82,
        cholesterol="Borderline (200-239 mg/dL)",
        glucose="Diabetes (>=126 mg/dL)",
        smoking="No", alcohol="Yes",
        cpt="Non-Anginal Pain", ecg="Left Ventricular Hypertrophy",
        max_hr=108, ex_angina="Yes", oldpeak=1.9, slope="Flat",
    ),
}

defaults = PRESET_VALUES.get(preset, dict(
    age=50, gender="Male", height_cm=170, weight_kg=75,
    systolic_bp=125, diastolic_bp=80,
    cholesterol="Borderline (200-239 mg/dL)",
    glucose="Normal (<100 mg/dL)",
    smoking="No", alcohol="No",
    cpt="Atypical Angina", ecg="Normal",
    max_hr=150, ex_angina="No", oldpeak=0.5, slope="Upsloping",
))


def _idx(options: dict, key: str) -> int:
    return list(options.keys()).index(defaults[key])


# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("patient_form"):
    # Row 1 - Demographics & body metrics
    st.markdown('<div class="section-head">1 - Demographics & Body Metrics</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age (years)", 20, 95, defaults["age"])
    with c2:
        gender_label = st.selectbox("Gender", list(GENDER_OPTS.keys()),
                                    index=_idx(GENDER_OPTS, "gender"))
    with c3:
        height_cm = st.number_input("Height (cm)", 140, 210, defaults["height_cm"])
    with c4:
        weight_kg = st.number_input("Weight (kg)", 35, 180, defaults["weight_kg"])

    # Row 2 - Vitals
    st.markdown('<div class="section-head">2 - Vital Signs</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        systolic_bp = st.number_input("Systolic BP (mmHg)", 85, 230,
                                      defaults["systolic_bp"])
    with c2:
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", 50, 140,
                                       defaults["diastolic_bp"])
    with c3:
        max_hr = st.number_input("Max Heart Rate (bpm)", 55, 210,
                                 defaults["max_hr"])

    # Row 3 - Labs & lifestyle
    st.markdown('<div class="section-head">3 - Labs & Lifestyle</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        chol_label = st.selectbox("Cholesterol level",
                                  list(CHOL_OPTS.keys()),
                                  index=_idx(CHOL_OPTS, "cholesterol"))
    with c2:
        gluc_label = st.selectbox("Fasting glucose",
                                  list(GLUC_OPTS.keys()),
                                  index=_idx(GLUC_OPTS, "glucose"))
    with c3:
        smoke_label = st.selectbox("Current smoker",
                                   list(YESNO_OPTS.keys()),
                                   index=_idx(YESNO_OPTS, "smoking"))
    with c4:
        alcohol_label = st.selectbox("Regular alcohol intake",
                                     list(YESNO_OPTS.keys()),
                                     index=_idx(YESNO_OPTS, "alcohol"))

    # Row 4 - ECG / Exercise findings
    st.markdown('<div class="section-head">4 - ECG & Exercise-Test Findings</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        cpt_label = st.selectbox("Chest pain type",
                                 list(CPT_OPTS.keys()),
                                 index=_idx(CPT_OPTS, "cpt"))
        ex_angina_label = st.selectbox("Exercise-induced angina",
                                       list(YESNO_OPTS.keys()),
                                       index=_idx(YESNO_OPTS, "ex_angina"))
    with c2:
        ecg_label = st.selectbox("Resting ECG result",
                                 list(ECG_OPTS.keys()),
                                 index=_idx(ECG_OPTS, "ecg"))
        slope_label = st.selectbox("ST-segment slope (peak exercise)",
                                   list(SLOPE_OPTS.keys()),
                                   index=_idx(SLOPE_OPTS, "slope"))

    c1, _ = st.columns([1, 3])
    with c1:
        oldpeak = st.number_input(
            "Oldpeak (ST depression, mm)", 0.0, 6.5,
            float(defaults["oldpeak"]), step=0.1, format="%.1f",
        )

    submitted = st.form_submit_button("Run Detection", type="primary",
                                      use_container_width=True)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def build_input_row() -> pd.DataFrame:
    """Assemble a single-row dataframe in the exact column order the model expects."""
    row = {
        "age": age,
        "gender": GENDER_OPTS[gender_label],
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "cholesterol_level": CHOL_OPTS[chol_label],
        "glucose_level": GLUC_OPTS[gluc_label],
        "smoking": YESNO_OPTS[smoke_label],
        "alcohol_intake": YESNO_OPTS[alcohol_label],
        "chest_pain_type": CPT_OPTS[cpt_label],
        "resting_ecg": ECG_OPTS[ecg_label],
        "max_heart_rate": max_hr,
        "exercise_angina": YESNO_OPTS[ex_angina_label],
        "oldpeak": oldpeak,
        "st_slope": SLOPE_OPTS[slope_label],
    }
    # Engineered features - same formulas as in the notebook's compute_engineered()
    row["bmi"] = row["weight_kg"] / (row["height_cm"] / 100.0) ** 2
    row["pulse_pressure"] = row["systolic_bp"] - row["diastolic_bp"]
    row["chronotropic_response_index"] = (
        row["max_heart_rate"] / (220.0 - row["age"])
    )
    return pd.DataFrame([row])[FEATURE_COLUMNS]


if submitted:
    X = build_input_row()

    proba = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(proba[pred_idx])

    # Prediction card
    card_class = PRED_CARD_CLASS[pred_class]
    st.markdown(
        f"""
        <div class="pred-card {card_class}">
            <h2>Prediction: {pred_class}</h2>
            <p>Model confidence: <b>{confidence*100:.1f}%</b> &nbsp;|&nbsp; {PRED_BLURB[pred_class]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 1])

    # Class probabilities
    with left:
        st.markdown('<div class="section-head">Class probabilities</div>',
                    unsafe_allow_html=True)
        proba_df = pd.DataFrame(
            {"Class": CLASS_NAMES, "Probability": proba}
        ).sort_values("Probability", ascending=False).reset_index(drop=True)

        st.bar_chart(proba_df.set_index("Class"), height=240)
        st.dataframe(
            proba_df.style.format({"Probability": "{:.4f}"}),
            use_container_width=True, hide_index=True,
        )

    # Derived clinical features
    with right:
        st.markdown('<div class="section-head">Derived clinical features</div>',
                    unsafe_allow_html=True)
        bmi = float(X["bmi"].iloc[0])
        pp = float(X["pulse_pressure"].iloc[0])
        cri = float(X["chronotropic_response_index"].iloc[0])

        def bmi_band(v: float) -> str:
            if v < 18.5: return "Underweight"
            if v < 25:   return "Normal"
            if v < 30:   return "Overweight"
            return "Obese"

        def pp_band(v: float) -> str:
            if v < 40: return "Low / Normal"
            if v <= 60: return "Borderline"
            return "Elevated"

        def cri_band(v: float) -> str:
            if v >= 0.85: return "Normal response"
            if v >= 0.70: return "Mildly reduced"
            return "Chronotropic incompetence"

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">BMI (kg/m²)</div>'
                f'<div class="metric-value">{bmi:.1f}</div>'
                f'<div class="metric-label">{bmi_band(bmi)}</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-box"><div class="metric-label">Pulse pressure (mmHg)</div>'
                f'<div class="metric-value">{pp:.0f}</div>'
                f'<div class="metric-label">{pp_band(pp)}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-box"><div class="metric-label">Chronotropic response index</div>'
            f'<div class="metric-value">{cri:.2f}</div>'
            f'<div class="metric-label">{cri_band(cri)} '
            f'(max HR / (220 - age))</div></div>',
            unsafe_allow_html=True,
        )

    # Submitted feature snapshot
    with st.expander("Show the full feature vector sent to the model"):
        snapshot = X.T.reset_index()
        snapshot.columns = ["Feature", "Value"]
        st.dataframe(snapshot, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="footnote">Disclaimer: this is a research prototype trained '
        'on synthetic data. Outputs must not be used for clinical decision-making.</div>',
        unsafe_allow_html=True,
    )
else:
    st.info("Fill in the form above and press **Run Detection** to score the patient.",
            icon="ℹ")
