import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="German Credit Risk", layout="wide", page_icon="🏦")

st.title("🏦 German Credit Risk Predictor")
st.markdown("**AI-Powered Credit Assessment | Logistic Regression + SMOTE**")

# ---------------- LOAD MODEL ----------------
@st.cache_data
def load_models():
    try:
        model_path = "models/credit_model.pkl"
        feature_path = "models/feature_info.pkl"

        if not os.path.exists(model_path) or not os.path.exists(feature_path):
            raise FileNotFoundError("Model files not found. Ensure 'models/' folder is present.")

        model = joblib.load(model_path)
        feature_info = joblib.load(feature_path)

        return model, feature_info

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()


model, feature_info = load_models()

preprocessor = feature_info['preprocessor']
num_cols = feature_info['num_cols']
cat_cols = feature_info['cat_cols']

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("📝 Applicant Details")
st.sidebar.markdown("---")

# Numeric inputs
col1, col2 = st.sidebar.columns(2)
with col1:
    age = st.number_input("👴 Age", 18, 80, 30)
with col2:
    duration = st.number_input("📅 Duration (months)", 1, 72, 12)

credit_amount = st.sidebar.number_input("💰 Credit Amount", 250, 20000, 5000)

# Categorical inputs (exact match)
sex_options = ['male', 'female']
job_options = ['unemp', 'unskilled', 'skilled', 'mgmt']
housing_options = ['own', 'rent', 'free']
savings_options = ['none', 'low', 'medium', 'high', 'unknown']
checking_options = ['none', 'low', 'medium', 'high', 'unknown']
purpose_options = [
    'radio/TV', 'education', 'furniture/equipment',
    'car', 'business', 'repayment', 'vacation/others'
]

sex = st.sidebar.selectbox("👤 Sex", sex_options)
job = st.sidebar.selectbox("💼 Job", job_options)
housing = st.sidebar.selectbox("🏠 Housing", housing_options)
savings = st.sidebar.selectbox("🏦 Saving accounts", savings_options)
checking = st.sidebar.selectbox("💳 Checking account", checking_options)
purpose = st.sidebar.selectbox("🎯 Purpose", purpose_options)

# ---------------- INPUT DATAFRAME ----------------
input_dict = {
    'Age': [age],
    'Credit amount': [credit_amount],
    'Duration': [duration],
    'Sex': [sex],
    'Job': [job],
    'Housing': [housing],
    'Saving accounts': [savings],
    'Checking account': [checking],
    'Purpose': [purpose]
}

input_df = pd.DataFrame(input_dict)

# ---------------- PREDICTION ----------------
if st.sidebar.button("🔍 Predict Credit Risk", type="primary"):
    try:
        with st.spinner("Analyzing credit risk..."):

            # Transform
            input_processed = preprocessor.transform(input_df)

            # Predict
            prediction = model.predict(input_processed)[0]
            probability = model.predict_proba(input_processed)[0, 1]

            # Layout
            col1, col2 = st.columns([1, 2])

            # ---- LEFT PANEL ----
            with col1:
                st.metric("Risk Score", f"{probability:.1%}")

                if probability > 0.5:
                    st.error("❌ HIGH RISK")
                else:
                    st.success("✅ LOW RISK")

            # ---- RIGHT PANEL (Gauge) ----
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probability,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Credit Risk Level"},
                    delta={'reference': 0.5},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.4], 'color': "green"},
                            {'range': [0.4, 0.7], 'color': "orange"},
                            {'range': [0.7, 1], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.5
                        }
                    }
                ))

                st.plotly_chart(fig, use_container_width=True)

            # ---- RISK SUMMARY ----
            st.subheader("📊 Risk Analysis")

            st.info(f"""
            **Profile Summary:**
            - Age: {age} years
            - Credit Amount: ₹{credit_amount:,}
            - Duration: {duration} months  
            - Job: {job.title()}
            - Risk Score: **{probability:.1%}**
            """)

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
        st.error("Check model compatibility or preprocessing pipeline.")

# ---------------- MODEL INFO ----------------
with st.sidebar.expander("📈 Model Performance"):
    st.success("✅ ROC-AUC: 78%+")
    st.success("✅ SMOTE balanced classes")
    st.success("✅ OneHotEncoder + Logistic Regression")
    st.info("German Credit Dataset (1000 samples)")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("*Professional Fintech Portfolio App | Deploy to Streamlit Cloud*")