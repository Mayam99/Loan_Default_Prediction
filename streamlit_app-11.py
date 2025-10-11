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

# Replace with your dataset's feature names
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
# Add all the required features here...

if st.button("Predict"):
    features = np.array([[feature1, feature2]])  # Adjust order to match training columns
    prediction = model.predict(features)
    prob = model.predict_proba(features)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Likely to Default. (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Safe Customer. (Probability: {prob:.2f})")
