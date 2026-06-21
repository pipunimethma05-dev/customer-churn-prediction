import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
import pandas as pd
import joblib

model  = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

expected_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

st.set_page_config(page_title="Churn Predictor", page_icon="📊")
st.title("📊 Customer Churn Prediction")
st.divider()

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

st.divider()

clicked = st.button("🔍 Predict Churn", use_container_width=True)


if clicked:
    total_charges = monthly_charges * tenure

    input_data = {
        'gender':           0 if gender == "Female" else 1,
        'SeniorCitizen':    1 if senior == "Yes" else 0,
        'Partner':          1 if partner == "Yes" else 0,
        'Dependents':       1 if dependents == "Yes" else 0,
        'tenure':           tenure,
        'PhoneService':     1,
        'MultipleLines':    0,
        'InternetService':  {"DSL": 0, "Fiber optic": 1, "No": 2}[internet],
        'OnlineSecurity':   0,
        'OnlineBackup':     0,
        'DeviceProtection': 0,
        'TechSupport':      {"No": 0, "No internet service": 1, "Yes": 2}[tech_support],
        'StreamingTV':      0,
        'StreamingMovies':  0,
        'Contract':         {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
        'PaperlessBilling': 1 if paperless == "Yes" else 0,
        'PaymentMethod':    {"Bank transfer (automatic)": 0, "Credit card (automatic)": 1, "Electronic check": 2, "Mailed check": 3}[payment],
        'MonthlyCharges':   monthly_charges,
        'TotalCharges':     total_charges,
    }

    input_df = pd.DataFrame([input_data])[expected_features]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.error("⚠️ This customer is likely to CHURN")
    else:
        st.success("✅ This customer is likely to STAY")

    st.metric("Churn Probability", f"{probability:.1%}")
    st.progress(float(probability))

    if probability >= 0.7:
        st.warning("🔴 High Risk")
    elif probability >= 0.4:
        st.warning("🟡 Medium Risk")
    else:
        st.info("🟢 Low Risk")