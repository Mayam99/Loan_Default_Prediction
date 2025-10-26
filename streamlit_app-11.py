import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    with open("logistic_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("💰 Loan Default Prediction App")
st.write("This app predicts if a customer will default on a loan using Logistic Regression.")

# --- User Inputs ---
st.header("Enter Customer Details:")
age = st.number_input("Age", min_value=18, max_value=100)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900)

# Replace with your dataset's feature names
age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)
income = st.number_input("Enter Annual Income", min_value=0.0, value=50000.0)
loan_amount = st.number_input("Enter Loan Amount", min_value=0.0, value=100000.0)
# Add all the required features here...

if st.button("Predict"):
    features = np.array([[age, credit_score, annual_income, loan_amount]])  # Adjust order to match training columns
    prediction = model.predict(features)
    prob = model.predict_proba(features)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Likely to Default. (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Safe Customer. (Probability: {prob:.2f})")



