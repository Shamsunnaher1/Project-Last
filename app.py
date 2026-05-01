import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dengue Predictor",
    page_icon="🦟",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #f0f4f8; }

section[data-testid="stSidebar"] {
    background: #1a202c;
    border-right: none;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label { color: #a0aec0 !important; font-size: 0.82rem; }
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select { background: #2d3748 !important; color: #e2e8f0 !important; border: 1px solid #4a5568 !important; }

.metric-card {
    background: white;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    margin-bottom: 1rem;
}

.page-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #1a202c;
    margin: 0;
    line-height: 1.15;
}

.page-sub {
    color: #718096;
    font-size: 0.95rem;
    margin-top: 0.3rem;
}

.section-head {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #a0aec0;
    margin-bottom: 0.8rem;
}

.result-positive {
    background: linear-gradient(135deg, #fff5f5, #fed7d7);
    border-left: 5px solid #e53e3e;
    border-radius: 12px;
    padding: 1.6rem 2rem;
}
.result-negative {
    background: linear-gradient(135deg, #f0fff4, #c6f6d5);
    border-left: 5px solid #38a169;
    border-radius: 12px;
    padding: 1.6rem 2rem;
}
.result-title { font-family: 'DM Serif Display', serif; font-size: 1.8rem; }
.result-pos-title { color: #c53030; }
.result-neg-title { color: #276749; }
.result-body { color: #4a5568; font-size: 0.92rem; margin-top: 0.4rem; }

.disclaimer {
    font-size: 0.75rem;
    color: #a0aec0;
    border-top: 1px solid #e2e8f0;
    padding-top: 0.8rem;
    margin-top: 1.5rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    artifact_err = str(e)

# ── Preprocessing constants (mirror training pipeline) ─────────────────────────
# IQR caps: (lower_bound, upper_bound)
IQR_CAPS = {
    "platelet_count":              (50_000, 400_000),
    "platelet_distribution_width": (8.0,    20.0),
    "hemoglobin_g_dl":             (8.0,    18.0),
}

LOG_COLS   = ["platelet_count", "wbc_count", "platelet_distribution_width"]
SCALE_COLS = ["age", "hemoglobin_g_dl", "wbc_count", "platelet_count", "platelet_distribution_width"]

FEATURE_ORDER = [
    "age", "hemoglobin_g_dl", "wbc_count", "differential_count",
    "rbc_count", "platelet_count", "platelet_distribution_width",
    "gender_Female", "gender_Male",
]

def preprocess(age, gender, hemoglobin, wbc, differential, rbc, platelet, pdw):
    d = {
        "age":                         float(age),
        "hemoglobin_g_dl":             float(hemoglobin),
        "wbc_count":                   float(wbc),
        "differential_count":          float(differential),
        "rbc_count":                   float(rbc),
        "platelet_count":              float(platelet),
        "platelet_distribution_width": float(pdw),
        # One-hot encode gender (drop_first=True drops 'Child')
        "gender_Female":               1.0 if gender == "Female" else 0.0,
        "gender_Male":                 1.0 if gender == "Male"   else 0.0,
    }

    # Step 1 — IQR capping
    for col, (lo, hi) in IQR_CAPS.items():
        d[col] = float(np.clip(d[col], lo, hi))

    # Step 2 — Log transformation
    for col in LOG_COLS:
        d[col] = float(np.log1p(d[col]))

    # Step 3 — Standard scaling
    scale_arr = np.array([[d[c] for c in SCALE_COLS]])
    scaled    = scaler.transform(scale_arr)[0]
    for i, c in enumerate(SCALE_COLS):
        d[c] = float(scaled[i])

    # Step 4 — Assemble final feature vector in model's expected order
    return np.array([[d[f] for f in FEATURE_ORDER]])


# ── Sidebar — user inputs ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦟 Patient Inputs")
    st.markdown("---")

    st.markdown("**Demographics**")
    age    = st.number_input("Age (years)",  min_value=0,   max_value=100,   value=25,      step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Child"])

    st.markdown("---")
    st.markdown("**Blood Panel**")
    hemoglobin   = st.number_input("Hemoglobin (g/dL)",              min_value=1.0,   max_value=25.0,    value=13.5,    step=0.1,  format="%.1f")
    wbc          = st.number_input("WBC Count (×10³/µL)",            min_value=0.1,   max_value=100.0,   value=6.0,     step=0.1,  format="%.1f")
    differential = st.number_input("Differential Count (%)",         min_value=0.0,   max_value=100.0,   value=60.0,    step=0.5,  format="%.1f")
    rbc          = st.number_input("RBC Count (×10⁶/µL)",           min_value=0.5,   max_value=10.0,    value=4.5,     step=0.01, format="%.2f")
    platelet     = st.number_input("Platelet Count (/µL)",           min_value=5_000, max_value=500_000, value=200_000, step=1_000)
    pdw          = st.number_input("Platelet Distribution Width (%)", min_value=1.0,  max_value=30.0,    value=12.0,    step=0.1,  format="%.1f")

    st.markdown("---")
    predict_btn = st.button("🔍 Run Prediction", use_container_width=True, type="primary")


# ── Main panel ─────────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">Dengue Fever<br>Risk Assessment</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">Fill in patient values on the left and click <strong>Run Prediction</strong>.</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if not artifacts_ok:
    st.error(
        f"❌ Could not load `model.pkl` / `scaler.pkl`. "
        f"Make sure both files are in the same directory as `app.py`.\n\n`{artifact_err}`"
    )
    st.stop()

# Summary metric cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="section-head">Age</div>'
                f'<b style="font-size:1.5rem">{age} yrs</b></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="section-head">Gender</div>'
                f'<b style="font-size:1.5rem">{gender}</b></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="section-head">Platelets (/µL)</div>'
                f'<b style="font-size:1.5rem">{platelet:,}</b></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="section-head">Hemoglobin (g/dL)</div>'
                f'<b style="font-size:1.5rem">{hemoglobin}</b></div>', unsafe_allow_html=True)

st.markdown("---")

# Result area
if predict_btn:
    with st.spinner("Analyzing…"):
        try:
            X          = preprocess(age, gender, hemoglobin, wbc, differential, rbc, platelet, pdw)
            prediction = model.predict(X)[0]
            proba      = model.predict_proba(X)[0]
            confidence = proba[int(prediction)] * 100

            if prediction == 1:
                st.markdown("""
                <div class="result-positive">
                    <div class="result-title result-pos-title">🔴 Dengue Positive</div>
                    <div class="result-body">
                        The model predicts a <strong>Positive</strong> dengue result.<br>
                        Immediate clinical evaluation and confirmatory testing is strongly advised.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-negative">
                    <div class="result-title result-neg-title">🟢 Dengue Negative</div>
                    <div class="result-body">
                        The model predicts a <strong>Negative</strong> dengue result.<br>
                        Continue monitoring if clinical symptoms persist.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            bar_col, _ = st.columns([2, 1])
            with bar_col:
                st.markdown('<div class="section-head">Model Confidence</div>', unsafe_allow_html=True)
                st.progress(int(confidence), text=f"{confidence:.1f}%")

        except Exception as e:
            st.error(f"Prediction error: {e}")
else:
    st.info("👈  Enter patient values in the sidebar and click **Run Prediction**.")

st.markdown(
    '<div class="disclaimer">⚠️ For research and screening purposes only. '
    'Not a substitute for professional medical diagnosis. Always consult a qualified clinician.</div>',
    unsafe_allow_html=True
)
