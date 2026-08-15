"""
Heart Disease Risk Prediction - Streamlit Application.
Developed during PYML Internship at Anveshan Foundation, IGDTUW.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.preprocessing import FEATURE_SCHEMA, NUMERICAL_COLS, CATEGORICAL_COLS
from src.predict import HeartDiseasePredictor
from src.evaluation import extract_xgboost_gain_importance, plot_confusion_matrix, plot_feature_importance


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Heart Disease Risk Prediction | ML Diagnostic Demo",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished medical ML styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a5568;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #fffaf0;
        border-left: 4px solid #dd6b20;
        padding: 0.85rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        font-size: 0.92rem;
        color: #7b341e;
    }
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .prediction-card-high {
        background-color: #fff5f5;
        border: 2px solid #e53e3e;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .prediction-card-low {
        background-color: #f0fff4;
        border: 2px solid #38a169;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .stButton>button {
        background-color: #2b6cb0;
        color: white;
        font-weight: 600;
        padding: 0.6rem 2rem;
        border-radius: 6px;
        border: none;
        width: 100%;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #2c5282;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load serialized predictor pipeline with caching."""
    return HeartDiseasePredictor(pipeline_path="models/xgboost_pipeline.pkl")


@st.cache_data
def load_metrics():
    """Load benchmark metrics JSON."""
    metrics_path = "models/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    # Sidebar Metadata
    st.sidebar.image("https://img.icons8.com/color/96/000000/heart-with-pulse.png", width=70)
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Go to Section:",
        ["🔍 Prediction & Risk Assessment", "📊 Model Performance & Benchmarks"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Project Origin:**")
    st.sidebar.caption("PYML Internship Research Project  \n**Anveshan Foundation, IGDTUW**")
    st.sidebar.markdown("**Model Engine:**")
    st.sidebar.caption("XGBoost Classifier (Leakage-safe ImbPipeline with 5-fold CV & SMOTE)")

    # Title & Global Educational Disclaimer
    st.markdown('<div class="main-header">❤️ Heart Disease Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Machine Learning–based cardiovascular risk prediction & comparative benchmark</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>Medical Disclaimer:</strong> This application is an educational machine-learning demonstration and 
        is <strong>not</strong> a medical diagnostic tool or a substitute for professional medical advice, clinical examination, 
        or physician judgment.
    </div>
    """, unsafe_allow_html=True)

    predictor = load_predictor()
    metrics_data = load_metrics()

    # ==========================================
    # SECTION 1: INFERENCE & RISK ASSESSMENT
    # ==========================================
    if app_mode == "🔍 Prediction & Risk Assessment":
        st.subheader("Patient Clinical Data Input")
        st.write("Enter patient demographics, vital signs, and diagnostic test results below:")

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("##### 👤 Patient Demographics")
                age = st.slider(
                    "Age (years)", 
                    min_value=20, max_value=90, value=54, step=1,
                    help="Patient age in years."
                )
                sex = st.selectbox(
                    "Sex", 
                    options=["M", "F"],
                    format_func=lambda x: "Male (M)" if x == "M" else "Female (F)",
                    help="Biological sex of the patient."
                )
                
                st.markdown("##### 🩸 Metabolic Indicator")
                fasting_bs = st.selectbox(
                    "Fasting Blood Sugar > 120 mg/dl",
                    options=[0, 1],
                    format_func=lambda x: "Yes (Fasting BS > 120 mg/dl)" if x == 1 else "No (Fasting BS ≤ 120 mg/dl)",
                    help="1 if fasting blood sugar > 120 mg/dl (hyperglycemia), 0 otherwise."
                )

            with col2:
                st.markdown("##### 🩺 Vital Measurements")
                resting_bp = st.slider(
                    "Resting Blood Pressure (mm Hg)", 
                    min_value=80, max_value=200, value=130, step=1,
                    help="Resting blood pressure measured in mm Hg on hospital admission."
                )
                cholesterol = st.slider(
                    "Serum Cholesterol (mm/dl)", 
                    min_value=0, max_value=600, value=220, step=5,
                    help="Serum cholesterol in mm/dl. Values near 0 represent missing measurements in clinical records."
                )
                max_hr = st.slider(
                    "Maximum Heart Rate Achieved", 
                    min_value=60, max_value=210, value=140, step=1,
                    help="Maximum heart rate achieved during exercise stress testing."
                )

            with col3:
                st.markdown("##### 🫀 Diagnostic / ECG Attributes")
                chest_pain_type = st.selectbox(
                    "Chest Pain Type",
                    options=["ASY", "NAP", "ATA", "TA"],
                    format_func=lambda x: {
                        "ASY": "ASY — Asymptomatic",
                        "NAP": "NAP — Non-Anginal Pain",
                        "ATA": "ATA — Atypical Angina",
                        "TA": "TA — Typical Angina"
                    }[x],
                    help="Nature of chest discomfort or pain reported by patient."
                )
                resting_ecg = st.selectbox(
                    "Resting Electrocardiogram (ECG)",
                    options=["Normal", "LVH", "ST"],
                    format_func=lambda x: {
                        "Normal": "Normal — Normal rhythm",
                        "LVH": "LVH — Left Ventricular Hypertrophy",
                        "ST": "ST — ST-T wave abnormality"
                    }[x],
                    help="Resting electrocardiogram results."
                )
                exercise_angina = st.selectbox(
                    "Exercise-Induced Angina",
                    options=["N", "Y"],
                    format_func=lambda x: "Yes (Angina induced by exercise)" if x == "Y" else "No (No exercise-induced angina)",
                    help="Whether exercise induces angina."
                )
                oldpeak = st.slider(
                    "ST Depression ('Oldpeak')", 
                    min_value=-2.0, max_value=6.0, value=1.0, step=0.1,
                    help="ST depression induced by exercise relative to rest (measured in mm)."
                )
                st_slope = st.selectbox(
                    "Peak Exercise ST Slope",
                    options=["Flat", "Up", "Down"],
                    format_func=lambda x: {
                        "Up": "Up — Upsloping ST segment",
                        "Flat": "Flat — Flat ST segment",
                        "Down": "Down — Downsloping ST segment"
                    }[x],
                    help="Slope of peak exercise ST segment."
                )

            submit_btn = st.form_submit_button("PREDICT RISK")

        if submit_btn:
            input_dict = {
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

            try:
                result = predictor.predict(input_dict)
                prob = result['model_estimated_probability']
                pred = result['prediction']
                label = result['prediction_label']

                st.markdown("---")
                st.subheader("Model Inference Output")

                if pred == 1:
                    st.markdown(f"""
                    <div class="prediction-card-high">
                        <h3 style="color: #c53030; margin-top: 0;">Predicted Class: {label}</h3>
                        <p style="font-size: 1.1rem; color: #2d3748; margin-bottom: 0.5rem;">
                            <strong>Model-Estimated Probability:</strong> <code>{prob*100:.1f}%</code>
                        </p>
                        <p style="color: #4a5568; font-size: 0.95rem; margin-bottom: 0;">
                            The XGBoost classifier identified elevated statistical patterns associated with cardiovascular risk 
                            (e.g., ST-Slope dynamics, chest pain profile, exercise-induced angina, and ST depression).
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card-low">
                        <h3 style="color: #276749; margin-top: 0;">Predicted Class: {label}</h3>
                        <p style="font-size: 1.1rem; color: #2d3748; margin-bottom: 0.5rem;">
                            <strong>Model-Estimated Probability:</strong> <code>{prob*100:.1f}%</code>
                        </p>
                        <p style="color: #4a5568; font-size: 0.95rem; margin-bottom: 0;">
                            The input profile exhibits health indicators typically observed among non-disease baseline cohorts.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")
                st.progress(float(prob))
                st.caption(f"Model-estimated probability score: {prob*100:.1f}% (Class 1 Likelihood)")

                st.markdown("##### 📌 Key Model Features Overview")
                st.info(
                    "The model uses clinical and demographic variables including age, serum cholesterol, "
                    "resting blood pressure, maximum heart rate achieved, ST depression (oldpeak), "
                    "exercise-induced angina, and ST slope characteristics to calculate this decision score."
                )

            except Exception as err:
                st.error(f"Inference error: {err}")

    # ==========================================
    # SECTION 2: MODEL PERFORMANCE & BENCHMARKS
    # ==========================================
    elif app_mode == "📊 Model Performance & Benchmarks":
        st.subheader("Model Performance & Comparative Benchmark")
        st.write(
            "Evaluation across six supervised classification algorithms trained using "
            "an 80/20 stratified split with 5-fold cross-validation and SMOTE balancing:"
        )

        if metrics_data:
            # Metrics comparison table
            table_rows = []
            for m_name, m in metrics_data.items():
                table_rows.append({
                    "Algorithm": m_name,
                    "Accuracy": f"{m['accuracy']*100:.2f}%",
                    "Precision": f"{m['precision']*100:.2f}%",
                    "Recall": f"{m['recall']*100:.2f}%",
                    "F1-Score": f"{m['f1_score']*100:.2f}%",
                    "ROC-AUC": f"{m['roc_auc']*100:.2f}%"
                })
            df_table = pd.DataFrame(table_rows)
            st.dataframe(df_table, use_container_width=True)

            st.info(
                "💡 **Model Selection Insight (XGBoost vs. KNN):** Both XGBoost and KNN achieved a tied peak test accuracy of **89.13%** "
                "on the held-out test split (n=184). While KNN demonstrated higher ROC-AUC (94.52%), **XGBoost** was selected for production "
                "because it delivered higher positive recall (91.18% vs 90.20%) and F1-score (90.29%), minimizing false-negative misses on high-risk patients, "
                "while providing tree gain-based feature interpretability."
            )

            st.markdown("---")
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.markdown("##### 📈 6-Classifier Metric Comparison")
                if os.path.exists("assets/model_comparison.png"):
                    st.image("assets/model_comparison.png", use_container_width=True)
                else:
                    st.caption("Benchmark chart available upon running training script.")

            with col_chart2:
                st.markdown("##### 🎯 XGBoost Test Confusion Matrix")
                if os.path.exists("assets/confusion_matrix_xgboost.png"):
                    st.image("assets/confusion_matrix_xgboost.png", use_container_width=True)
                else:
                    st.caption("Confusion matrix available upon running training script.")

            st.markdown("---")
            st.markdown("##### 🔍 XGBoost Gain-Based Feature Importance")
            st.write(
                "Gain reflects the relative contribution of each transformed feature to each decision tree split. "
                "Higher gain indicates greater predictive value in reducing loss."
            )
            
            try:
                df_imp = extract_xgboost_gain_importance(predictor.pipeline)
                col_imp1, col_imp2 = st.columns([3, 2])
                with col_imp1:
                    fig_imp = plot_feature_importance(df_imp, top_n=10)
                    st.pyplot(fig_imp)
                with col_imp2:
                    st.markdown("**Top Transformed Features by Gain:**")
                    st.dataframe(
                        df_imp[['CleanFeature', 'Gain', 'RelativeGain']].head(8).rename(
                            columns={'CleanFeature': 'Feature', 'Gain': 'Gain Score', 'RelativeGain': 'Weight Share'}
                        ).style.format({'Gain Score': '{:.2f}', 'Weight Share': '{:.1%}'}),
                        use_container_width=True
                    )
            except Exception as e:
                st.caption(f"Could not load dynamic feature importance: {e}")
                if os.path.exists("assets/feature_importance.png"):
                    st.image("assets/feature_importance.png", use_container_width=True)


if __name__ == "__main__":
    main()
