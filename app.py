import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the model
model = joblib.load("epilepsy_model.pkl")

# ------------------------------
# Top Section: Epilepsy Info
# ------------------------------
st.title("🧠 SeizureNet - Epilepsy Seizure Detection")
st.markdown("""
Epilepsy is a neurological disorder that affects brain activity, causing seizures.
Our app analyzes EEG signals to help detect the likelihood of seizure activity.

📌 **How it works:** Upload a 1-row CSV file with 179 EEG features.
""")

st.divider()

# ------------------------------
# File Upload
# ------------------------------
st.header("Upload EEG Data")
uploaded_file = st.file_uploader("Upload a CSV file (1 row, 179 columns)", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, header=None)

        if data.shape == (1, 179):
            proba = model.predict_proba(data)[0]
            prediction = np.argmax(proba)

            result = "⚠️ Seizure Detected" if prediction == 1 else "✅ No Seizure"
            st.subheader("Prediction Result:")
            st.success(result)

            # Show confidence
            st.subheader("Confidence Score:")
            st.write(f"Seizure: `{proba[1]*100:.2f}%`, No Seizure: `{proba[0]*100:.2f}%`")

            # Plot confidence chart
            st.subheader("Confidence Chart:")
            fig, ax = plt.subplots()
            ax.barh(["No Seizure", "Seizure"], proba, color=["green", "red"])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            st.pyplot(fig)

        else:
            st.error("CSV must have exactly 1 row and 179 columns.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
