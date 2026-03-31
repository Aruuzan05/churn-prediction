import os
import sys
import joblib
import pandas as pd
import streamlit as st

# Robust project root import setup
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import preprocess_features


@st.cache_resource
def load_artifacts():
    model_dir = os.path.join(PROJECT_ROOT, "data", "models")
    model = joblib.load(os.path.join(model_dir, "churn_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(model_dir, "feature_columns.pkl"))
    return model, scaler, feature_columns


st.title("Customer Churn Predictor")

model, scaler, feature_columns = load_artifacts()

# User inputs (raw schema)
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)

phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=65.0)
total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

if st.button("Predict churn risk"):
    raw_input = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])

    X = preprocess_features(raw_input)

    # Align to training features
    X = X.reindex(columns=feature_columns, fill_value=0)

    X_sc = scaler.transform(X)
    prob = model.predict_proba(X_sc)[0, 1]
    pred = int(prob >= 0.5)

    st.metric("Churn probability", f"{prob:.1%}")
    st.write("Prediction:", "Churn" if pred == 1 else "No Churn")