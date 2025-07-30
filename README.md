# SeizureNet 🧠⚡
*AI/ML-Based Epilepsy Seizure Detection and Prediction*

> A lightweight and accessible solution using EEG data and machine learning for proactive seizure monitoring.

## 📝 Project Overview

**SeizureNet** is a machine learning-based system designed to detect and predict epileptic seizures using preprocessed EEG (electroencephalogram) data. The project includes a trained Random Forest model and a minimal web interface (via Streamlit) for real-time prediction.

This project is part of an internship under the IEEE EMBS Pune Chapter, mentored by **Dr. Monika Dangore**.

---

## 🎯 Objectives

- Detect and predict seizure activity using AI/ML models.
- Provide a non-invasive and real-time prediction tool.
- Deliver a simple, user-friendly web interface for EEG data upload and prediction.

---

## 🧪 Dataset

- **Source**: Publicly available Kaggle EEG datasets (not MIT).
- **Format**: CSV files with 179 features per EEG segment.
- **Data Preprocessing**: Normalization, noise filtering, segmentation.

---

## 🧰 Tech Stack

| Component | Technology |
|----------|-------------|
| Language | Python |
| ML Libraries | `scikit-learn`, `NumPy`, `Pandas` |
| Visualization | `Matplotlib`, `Seaborn` |
| Interface | `Streamlit` |
| Dev Platform | Google Colab (model dev), Local (interface) |
| Version Control | Git + GitHub |

---

## 🧠 Model

- **Type**: Random Forest Classifier
- **Features Used**:
  - Statistical (Mean, Variance)
  - Frequency (PSD, FFT)
  - Complexity (Entropy, Energy)
- **Performance**:
  - Accuracy: **97.3%**
  - Precision: **95.6%**
  - Recall: **96.8%**
  - F1 Score: **96.2%**

---

## 🌐 Web Interface (Streamlit)

A minimal interface for:
- Uploading `.csv` EEG data files.
- Displaying prediction: **"Seizure"** or **"No Seizure"**.
- Real-time processing under 1 second (for small files).

---

## 🚀 Getting Started

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/seizurenet.git
    cd seizurenet
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Streamlit app**:
    ```bash
    streamlit run app.py
    ```

4. **Upload EEG CSV file** and view predictions.

---

## 📂 Repository Structure

