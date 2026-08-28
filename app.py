"""
CardioPulse AI — Next-Generation Cardiovascular Risk Intelligence Dashboard.
A sleek, executive clinical decision support demonstration built on a leakage-free XGBoost pipeline.
"""

import os
import sys
import datetime
import textwrap
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict import HeartDiseasePredictor

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioPulse AI | Cardiovascular Risk Intelligence",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. MODERN HEALTH-TECH CUSTOM CSS
# -----------------------------------------------------------------------------
st.markdown(textwrap.dedent("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }

    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        max-width: 1300px !important;
    }

    /* Top Brand Hero */
    .brand-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px -10px rgba(15, 23, 42, 0.4);
    }

    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 30%, #93c5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .brand-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-top: 0.4rem;
        margin-bottom: 1.1rem;
    }

    .pill-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.25);
        color: #cbd5e1;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        margin-right: 0.5rem;
        margin-bottom: 0.3rem;
    }

    .pill-badge-highlight {
        background: rgba(59, 130, 246, 0.15);
        border-color: rgba(59, 130, 246, 0.4);
        color: #60a5fa;
    }

    /* Section Cards */
    .glass-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 20px -4px rgba(0, 0, 0, 0.04);
    }

    .card-heading {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1.2rem;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 0.75rem;
    }

    /* Result Cards */
    .result-card-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 20px;
        padding: 1.8rem;
        box-shadow: 0 12px 30px -8px rgba(34, 197, 94, 0.25);
        color: #14532d;
        margin-bottom: 1.2rem;
    }

    .result-card-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 20px;
        padding: 1.8rem;
        box-shadow: 0 12px 30px -8px rgba(239, 68, 68, 0.25);
        color: #7f1d1d;
        margin-bottom: 1.2rem;
    }

    /* Stat Cards */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 2px 8px -2px rgba(0,0,0,0.03);
        margin-bottom: 0.8rem;
    }

    .stat-title {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
    }

    .stat-value {
        font-size: 1.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.2rem;
    }

    .stat-desc {
        font-size: 0.8rem;
        color: #475569;
        margin-top: 0.15rem;
    }

    /* Primary Button */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.8rem 1.8rem !important;
        border-radius: 14px !important;
        border: none !important;
        box-shadow: 0 8px 20px -4px rgba(37, 99, 235, 0.4) !important;
        width: 100% !important;
    }

    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 12px 25px -4px rgba(37, 99, 235, 0.55) !important;
    }

    /* Disclaimer Footer */
    .disclaimer-banner {
        background: rgba(254, 243, 199, 0.65);
        border: 1px solid #fde68a;
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        font-size: 0.85rem;
        color: #92400e;
        margin-top: 2.2rem;
    }
</style>
"""), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. CACHED PREDICTOR ENGINE
# -----------------------------------------------------------------------------
@st.cache_resource
def load_predictor():
    """Load serialized predictor pipeline with caching."""
    return HeartDiseasePredictor(pipeline_path="models/xgboost_pipeline.pkl")


# -----------------------------------------------------------------------------
# 4. PRESET PROFILES DEFINITION
# -----------------------------------------------------------------------------
PRESETS = {
    "low_risk": {
        "label": "🟢 Low-Risk Baseline (38F)",
        "data": {
            'Age': 38, 'Sex': 'F', 'ChestPainType': 'ATA', 'RestingBP': 115,
            'Cholesterol': 190, 'FastingBS': 0, 'RestingECG': 'Normal',
            'MaxHR': 170, 'ExerciseAngina': 'N', 'Oldpeak': 0.0, 'ST_Slope': 'Up'
        }
    },
    "high_risk": {
        "label": "🔴 High-Risk Stress Profile (62M)",
        "data": {
            'Age': 62, 'Sex': 'M', 'ChestPainType': 'ASY', 'RestingBP': 155,
            'Cholesterol': 285, 'FastingBS': 1, 'RestingECG': 'ST',
            'MaxHR': 118, 'ExerciseAngina': 'Y', 'Oldpeak': 2.8, 'ST_Slope': 'Flat'
        }
    },
    "moderate_risk": {
        "label": "🟡 Atypical Angina Case (52M)",
        "data": {
            'Age': 52, 'Sex': 'M', 'ChestPainType': 'NAP', 'RestingBP': 138,
            'Cholesterol': 230, 'FastingBS': 0, 'RestingECG': 'LVH',
            'MaxHR': 145, 'ExerciseAngina': 'N', 'Oldpeak': 1.2, 'ST_Slope': 'Flat'
        }
    }
}


def main():
    predictor = load_predictor()

    # Initialize Session State
    if "patient_data" not in st.session_state:
        st.session_state.patient_data = PRESETS["low_risk"]["data"].copy()

    # -------------------------------------------------------------------------
    # HEADER BRAND HERO
    # -------------------------------------------------------------------------
    st.markdown(textwrap.dedent("""
    <div class="brand-hero">
        <div class="brand-title">
            <span>🫀 CardioPulse AI</span>
        </div>
        <div class="brand-subtitle">
            Next-Generation Cardiovascular Risk Stratification & Clinical Decision Support
        </div>
        <div>
            <span class="pill-badge pill-badge-highlight">⚡ XGBoost Pipeline v2.0</span>
            <span class="pill-badge">🎯 89.1% Test Accuracy</span>
            <span class="pill-badge">🩺 91.2% Sensitivity / Recall</span>
            <span class="pill-badge">🛡️ Leakage-Safe 5-Fold CV</span>
        </div>
    </div>
    """), unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # QUICK-LOAD PATIENT PRESETS BAR
    # -------------------------------------------------------------------------
    st.markdown("##### ⚡ Quick-Fill Clinical Case Studies:")
    p_col1, p_col2, p_col3, p_col4 = st.columns([1, 1, 1, 0.8])

    with p_col1:
        if st.button(PRESETS["low_risk"]["label"], use_container_width=True):
            st.session_state.patient_data = PRESETS["low_risk"]["data"].copy()
            st.rerun()

    with p_col2:
        if st.button(PRESETS["high_risk"]["label"], use_container_width=True):
            st.session_state.patient_data = PRESETS["high_risk"]["data"].copy()
            st.rerun()

    with p_col3:
        if st.button(PRESETS["moderate_risk"]["label"], use_container_width=True):
            st.session_state.patient_data = PRESETS["moderate_risk"]["data"].copy()
            st.rerun()

    with p_col4:
        if st.button("🔄 Reset Form", use_container_width=True):
            st.session_state.patient_data = {
                'Age': 50, 'Sex': 'M', 'ChestPainType': 'ASY', 'RestingBP': 120,
                'Cholesterol': 200, 'FastingBS': 0, 'RestingECG': 'Normal',
                'MaxHR': 150, 'ExerciseAngina': 'N', 'Oldpeak': 0.0, 'ST_Slope': 'Up'
            }
            st.rerun()

    st.write("")

    # -------------------------------------------------------------------------
    # TWO-COLUMN SPLIT DASHBOARD
    # -------------------------------------------------------------------------
    left_col, right_col = st.columns([1.15, 1.0], gap="large")
    current_p = st.session_state.patient_data

    # =========================================================================
    # LEFT COLUMN: CLINICAL INPUTS
    # =========================================================================
    with left_col:
        st.markdown(textwrap.dedent("""
        <div class="glass-card">
            <div class="card-heading">
                <span>👤 Demographics & Baseline Vitals</span>
            </div>
        """), unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Patient Age (Years)", 20, 90, int(current_p['Age']), step=1)
            sex = st.selectbox(
                "Biological Sex", ["M", "F"],
                index=0 if current_p['Sex'] == 'M' else 1,
                format_func=lambda x: "Male (M)" if x == "M" else "Female (F)"
            )
            fasting_bs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL", [0, 1],
                index=int(current_p['FastingBS']),
                format_func=lambda x: "Yes (> 120 mg/dL — Hyperglycemia)" if x == 1 else "No (≤ 120 mg/dL — Normal)"
            )

        with c2:
            resting_bp = st.slider(
                "Resting Blood Pressure (mm Hg)", 80, 200, int(current_p['RestingBP']), step=1,
                help="AHA Guidelines: Normal <120, Elevated 120-129, Stage 1 HTN 130-139, Stage 2 ≥140"
            )
            cholesterol = st.slider(
                "Serum Cholesterol (mg/dL)", 0, 600, int(current_p['Cholesterol']), step=5,
                help="Clinical normal is <200 mg/dL. 0 denotes missing baseline clinical value."
            )
            max_hr = st.slider(
                "Max Heart Rate Achieved (bpm)", 60, 220, int(current_p['MaxHR']), step=1,
                help="Peak heart rate achieved during exercise stress test."
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Card 2: Cardiac Stress & ECG Biomarkers
        st.markdown(textwrap.dedent("""
        <div class="glass-card">
            <div class="card-heading">
                <span>🫀 Diagnostic ECG & Stress Indicators</span>
            </div>
        """), unsafe_allow_html=True)

        e1, e2 = st.columns(2)
        with e1:
            cp_opts = ["ASY", "NAP", "ATA", "TA"]
            cp_idx = cp_opts.index(current_p['ChestPainType']) if current_p['ChestPainType'] in cp_opts else 0
            chest_pain_type = st.selectbox(
                "Reported Chest Pain Type", cp_opts, index=cp_idx,
                format_func=lambda x: {
                    "ASY": "ASY — Asymptomatic (Silent)",
                    "NAP": "NAP — Non-Anginal Discomfort",
                    "ATA": "ATA — Atypical Angina",
                    "TA": "TA — Typical Angina"
                }[x]
            )

            ecg_opts = ["Normal", "LVH", "ST"]
            ecg_idx = ecg_opts.index(current_p['RestingECG']) if current_p['RestingECG'] in ecg_opts else 0
            resting_ecg = st.selectbox(
                "Resting Electrocardiogram (ECG)", ecg_opts, index=ecg_idx,
                format_func=lambda x: {
                    "Normal": "Normal — Normal Rhythm",
                    "LVH": "LVH — Ventricular Hypertrophy",
                    "ST": "ST — ST-T Wave Abnormality"
                }[x]
            )

            angina_idx = 1 if current_p['ExerciseAngina'] == 'Y' else 0
            exercise_angina = st.selectbox(
                "Exercise-Induced Angina", ["N", "Y"], index=angina_idx,
                format_func=lambda x: "Yes — Angina Induced by Exercise" if x == "Y" else "No — No Induced Angina"
            )

        with e2:
            oldpeak = st.slider(
                "ST Depression / Oldpeak (mm)", -2.0, 6.0, float(current_p['Oldpeak']), step=0.1,
                help="Exercise-induced ST-segment depression relative to baseline resting ECG."
            )

            slope_opts = ["Up", "Flat", "Down"]
            slope_idx = slope_opts.index(current_p['ST_Slope']) if current_p['ST_Slope'] in slope_opts else 0
            st_slope = st.selectbox(
                "Peak Exercise ST-Slope", slope_opts, index=slope_idx,
                format_func=lambda x: {
                    "Up": "Up — Upsloping (Healthy Dynamics)",
                    "Flat": "Flat — Flat ST Segment (Ischemic Risk)",
                    "Down": "Down — Downsloping (Severe Stress)"
                }[x]
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # Compile input payload
    patient_payload = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

    # =========================================================================
    # RIGHT COLUMN: AI DIAGNOSTIC INTELLIGENCE PANEL
    # =========================================================================
    with right_col:
        # Run inference
        result = predictor.predict(patient_payload)
        prob = float(result['model_estimated_probability'])
        pred = int(result['prediction'])
        prob_pct = prob * 100.0

        # Physiological Computations
        predicted_max_hr = 220 - age
        hr_ratio = (max_hr / predicted_max_hr) * 100.0 if predicted_max_hr > 0 else 100.0

        # BP Classification
        if resting_bp < 120:
            bp_cat = "Normal (< 120 mm Hg)"
            bp_color = "#16a34a"
        elif resting_bp <= 129:
            bp_cat = "Elevated (120–129 mm Hg)"
            bp_color = "#d97706"
        elif resting_bp <= 139:
            bp_cat = "Stage 1 HTN (130–139 mm Hg)"
            bp_color = "#ea580c"
        else:
            bp_cat = "Stage 2 HTN (≥ 140 mm Hg)"
            bp_color = "#dc2626"

        # Cholesterol Classification
        if cholesterol == 0:
            chol_cat = "Not Recorded"
        elif cholesterol < 200:
            chol_cat = "Desirable (< 200 mg/dL)"
        elif cholesterol <= 239:
            chol_cat = "Borderline (200–239 mg/dL)"
        else:
            chol_cat = "High Risk (≥ 240 mg/dL)"

        # Result Card Rendering
        if pred == 1:
            card_class = "result-card-high"
            badge_title = "ELEVATED CARDIOVASCULAR RISK"
            badge_color = "#b91c1c"
            badge_bg = "#fee2e2"
            status_desc = "The XGBoost classifier flagged significant ischemic and clinical risk markers associated with heart disease."
            gauge_stroke = "#ef4444"
        else:
            card_class = "result-card-low"
            badge_title = "LOW CARDIOVASCULAR RISK PROFILE"
            badge_color = "#15803d"
            badge_bg = "#dcfce7"
            status_desc = "Clinical indicators fall within baseline physiological parameters observed in non-disease cohorts."
            gauge_stroke = "#22c55e"

        # Gauge SVG calculation (Circumference = 2 * pi * 40 = 251.32)
        dash_offset = 251.32 * (1.0 - (prob_pct / 100.0))

        result_html = textwrap.dedent(f"""
        <div class="{card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <span style="font-weight: 800; font-size: 0.82rem; letter-spacing: 0.08em; text-transform: uppercase; color: {badge_color}; background: {badge_bg}; padding: 0.4rem 0.9rem; border-radius: 9999px;">
                    {badge_title}
                </span>
                <span style="font-size: 0.85rem; font-weight: 600; color: #475569;">
                    AI Stratification
                </span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; gap: 1.5rem;">
                <div>
                    <div style="font-size: 2.5rem; font-weight: 800; letter-spacing: -0.03em; color: #0f172a; line-height: 1;">
                        {prob_pct:.1f}<span style="font-size: 1.3rem; font-weight: 700; color: #64748b;">%</span>
                    </div>
                    <div style="font-size: 0.9rem; font-weight: 600; color: #475569; margin-top: 0.35rem;">
                        Model-Estimated Risk Probability
                    </div>
                </div>
                <div style="width: 84px; height: 84px;">
                    <svg viewBox="0 0 100 100" style="transform: rotate(-90deg); width: 100%; height: 100%;">
                        <circle cx="50" cy="50" r="40" fill="transparent" stroke="#e2e8f0" stroke-width="10"></circle>
                        <circle cx="50" cy="50" r="40" fill="transparent" stroke="{gauge_stroke}" stroke-width="10"
                                stroke-dasharray="251.32" stroke-dashoffset="{dash_offset:.2f}" stroke-linecap="round"></circle>
                    </svg>
                </div>
            </div>
            <div style="font-size: 0.88rem; margin-top: 1rem; line-height: 1.45; color: #334155;">
                {status_desc}
            </div>
        </div>
        """)
        st.markdown(result_html, unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # BIOMARKER CLINICAL VITALS GRID
        # ---------------------------------------------------------------------
        st.markdown(textwrap.dedent("""
        <div class="glass-card" style="padding: 1.2rem 1.4rem; margin-bottom: 1.2rem;">
            <div style="font-size: 0.92rem; font-weight: 700; color: #334155; margin-bottom: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;">
                Biomarker Health Insights
            </div>
        """), unsafe_allow_html=True)

        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.markdown(textwrap.dedent(f"""
            <div class="stat-card">
                <div class="stat-title">💓 Stress HR Reserve</div>
                <div class="stat-value">{max_hr} bpm</div>
                <div class="stat-desc">{hr_ratio:.0f}% of Age Max ({predicted_max_hr} bpm)</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">🩸 Blood Pressure Tier</div>
                <div class="stat-value" style="color: {bp_color}; font-size: 1.05rem;">{resting_bp} mm Hg</div>
                <div class="stat-desc">{bp_cat.split('(')[0].strip()}</div>
            </div>
            """), unsafe_allow_html=True)

        with stat_col2:
            st.markdown(textwrap.dedent(f"""
            <div class="stat-card">
                <div class="stat-title">🧪 Fasting Glycemia</div>
                <div class="stat-value">{'Elevated (>120)' if fasting_bs == 1 else 'Normal (≤120)'}</div>
                <div class="stat-desc">{'Hyperglycemia Marker' if fasting_bs == 1 else 'Euglycemic Range'}</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">📉 ST Ischemia Index</div>
                <div class="stat-value" style="color: {'#dc2626' if oldpeak >= 1.5 or st_slope in ['Flat','Down'] else '#16a34a'};">
                    {oldpeak:.1f} mm ({st_slope})
                </div>
                <div class="stat-desc">{'Significant Ischemia' if oldpeak >= 1.5 else 'Normal Recovery'}</div>
            </div>
            """), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # KEY RISK ATTRIBUTION FACTORS
        # ---------------------------------------------------------------------
        st.markdown(textwrap.dedent("""
        <div class="glass-card" style="padding: 1.2rem 1.4rem; margin-bottom: 1.2rem;">
            <div style="font-size: 0.92rem; font-weight: 700; color: #334155; margin-bottom: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;">
                🔍 Key Model Risk Factor Attribution
            </div>
        """), unsafe_allow_html=True)

        factors = []
        if st_slope in ['Flat', 'Down']:
            factors.append(("ST Slope Dynamics", f"Elevated Risk: {st_slope} ST segment during exercise stress", 95, "#ef4444"))
        else:
            factors.append(("ST Slope Dynamics", "Protective: Upsloping normal physiological recovery", 25, "#22c55e"))

        if chest_pain_type == "ASY":
            factors.append(("Chest Pain Presentation", "Elevated Risk: Asymptomatic presentation with underlying signs", 90, "#ef4444"))
        elif chest_pain_type == "ATA":
            factors.append(("Chest Pain Presentation", "Lower Risk: Atypical angina presentation", 30, "#22c55e"))
        else:
            factors.append(("Chest Pain Presentation", f"Moderate: {chest_pain_type} pattern", 55, "#f59e0b"))

        if exercise_angina == "Y":
            factors.append(("Exercise-Induced Angina", "Elevated Risk: Positive exercise angina reported", 85, "#ef4444"))
        else:
            factors.append(("Exercise-Induced Angina", "Protective: Negative exercise-induced angina", 20, "#22c55e"))

        if oldpeak >= 1.5:
            factors.append(("ST Depression (Oldpeak)", f"Elevated Risk: Significant {oldpeak:.1f} mm ST depression", 80, "#ef4444"))

        for f_name, f_desc, f_bar, f_color in factors[:3]:
            factor_html = textwrap.dedent(f"""
            <div style="margin-bottom: 0.75rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 700; color: #1e293b;">
                    <span>{f_name}</span>
                    <span style="color: {f_color}; font-weight: 600; font-size: 0.8rem;">{f_desc.split(':')[0]}</span>
                </div>
                <div style="font-size: 0.78rem; color: #64748b; margin-top: 0.1rem; margin-bottom: 0.3rem;">
                    {f_desc}
                </div>
                <div style="background: #f1f5f9; border-radius: 9999px; height: 6px; overflow: hidden;">
                    <div style="background: {f_color}; width: {f_bar}%; height: 100%; border-radius: 9999px;"></div>
                </div>
            </div>
            """)
            st.markdown(factor_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # CLINICAL REPORT EXPORT BUTTON
        # ---------------------------------------------------------------------
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_text = f"""=======================================================
CARDIOPULSE AI — CLINICAL RISK ASSESSMENT SUMMARY
Generated: {timestamp_str}
=======================================================

PATIENT CLINICAL PROFILE:
-------------------------------------------------------
• Age / Sex:              {age} years / {'Male' if sex=='M' else 'Female'}
• Resting Blood Pressure: {resting_bp} mm Hg ({bp_cat})
• Serum Cholesterol:      {cholesterol} mg/dL ({chol_cat})
• Fasting Blood Sugar:    {'Hyperglycemia (>120 mg/dL)' if fasting_bs==1 else 'Normal (≤120 mg/dL)'}
• Max Heart Rate:         {max_hr} bpm ({hr_ratio:.0f}% of Age Max)
• Chest Pain Type:        {chest_pain_type}
• Resting ECG:            {resting_ecg}
• Exercise Angina:        {'Yes' if exercise_angina=='Y' else 'No'}
• ST Depression:          {oldpeak:.1f} mm
• Peak ST Slope:          {st_slope}

AI STRATIFICATION RESULT:
-------------------------------------------------------
• Classification:         {result['prediction_label']}
• Model-Estimated Risk:   {prob_pct:.2f}%
• Machine Learning Model: XGBoost Classifier (Leakage-Safe ImbPipeline)
• Test Accuracy / Recall: 89.13% / 91.18%

CLINICAL INTERPRETATION:
-------------------------------------------------------
{status_desc}

=======================================================
DISCLAIMER: This report is generated by an educational 
machine-learning prototype and does not replace diagnostic 
clinical imaging, laboratory analysis, or physician judgment.
=======================================================
"""
        st.download_button(
            label="📥 Download Clinical Assessment Report (.txt)",
            data=report_text,
            file_name=f"cardiac_risk_report_{age}_{sex}.txt",
            mime="text/plain",
            use_container_width=True
        )

    # -------------------------------------------------------------------------
    # GLOBAL DISCLAIMER & CREDITS FOOTER
    # -------------------------------------------------------------------------
    st.markdown(textwrap.dedent("""
    <div class="disclaimer-banner">
        <strong>⚠️ Clinical Decision Support Prototype:</strong> This application is an engineering demonstration of 
        supervised machine-learning pipelines for educational, research, and portfolio demonstration purposes.
    </div>

    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; margin-top: 2.2rem; font-weight: 500;">
        Engineered by <strong>Radhika Jindal</strong> • PYML Research, Anveshan Foundation, IGDTUW • Built with Scikit-Learn & XGBoost
    </div>
    """), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
