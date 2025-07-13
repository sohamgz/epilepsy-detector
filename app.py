import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap


model = joblib.load("epilepsy_model.pkl")


st.title("🧠 SeizureNet - Epilepsy Seizure Detection")
st.markdown("""
Epilepsy is a neurological disorder that affects brain activity, causing seizures.
Our app analyzes EEG signals to help detect the likelihood of seizure activity.

📌 **How it works:** Upload a 1-row CSV file with 179 EEG features.
""")

st.divider()


st.header("Upload EEG Data")
uploaded_file = st.file_uploader("Upload a CSV file (1 row, 179 columns)", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, header=None)

        if data.shape == (1, 179):
            prediction = model.predict(data)[0]
            result = "⚠️ Seizure Detected" if prediction == 1 else "✅ No Seizure"
            st.subheader("Prediction Result:")
            st.success(result)

            # SHAP explanation
            st.subheader("Feature Contribution (SHAP)")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data)

            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.summary_plot(shap_values[1], data, show=False)
            st.pyplot(bbox_inches='tight')

        else:
            st.error("CSV must have exactly 1 row and 179 columns.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
