import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Load the trained model safely
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("logistic_model.pkl")
    return model

model = load_model()

st.title("💰 Loan Default Prediction App")
st.write("This app predicts if a customer will default on a loan using Logistic Regression.")

# -------------------------------
# User Inputs
# -------------------------------
st.header("Enter Customer Details:")

age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
annual_income = st.number_input("Enter Annual Income (₹)", min_value=0.0, value=50000.0)
loan_amount = st.number_input("Enter Loan Amount (₹)", min_value=0.0, value=100000.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    # Arrange features in same order as model was trained
    features = np.array([[age, credit_score, annual_income, loan_amount]])

    # Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    # Display output
    if prediction[0] == 1:
        st.error(f"⚠️ The customer is likely to **default** on the loan.\n\nDefault Probability: {probability:.2f}")
    else:
        st.success(f"✅ The customer is **safe** and unlikely to default.\n\nDefault Probability: {probability:.2f}")
