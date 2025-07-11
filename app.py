import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("epilepsy_model.pkl")

st.title("🧠 Epilepsy Seizure Detection")
st.write("Upload a CSV file with exactly one row of 178 EEG values.")

uploaded_file = st.file_uploader("Upload EEG data (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, header=None)
        
        if data.shape == (1, 179):
            prediction = model.predict(data)
            result = "⚠️ Seizure Detected" if prediction[0] == 1 else "✅ No Seizure"
            st.success(result)
        else:
            st.error("CSV must have exactly 1 row and 178 columns.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
