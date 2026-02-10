import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="Wine Classification", layout="wide")
st.title("🍷 Wine Classification – ML Models")

# ---------------------------
# Model paths
# ---------------------------
MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("⚙️ Controls")

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_PATHS.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

# ---------------------------
# Load Model
# ---------------------------
with open(MODEL_PATHS[selected_model_name], "rb") as f:
    model = pickle.load(f)

# ---------------------------
# Load Metrics
# ---------------------------
metrics_df = pd.read_csv("model/model_metrics.csv")

st.subheader("📊 Model Evaluation Metrics")
st.dataframe(metrics_df, use_container_width=True)

# ---------------------------
# Dataset Upload & Prediction
# ---------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "target" not in data.columns:
        st.error("❌ Uploaded CSV must contain a 'target' column")
        st.stop()

    X = data.drop("target", axis=1)
    y_true = data["target"]

    # Prediction
    y_pred = model.predict(X)

    # ---------------------------
    # Metrics
    # ---------------------------
    st.subheader("✅ Test Dataset Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
    col2.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.4f}")
    col3.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.4f}")

    col4, col5 = st.columns(2)
    col4.metric("F1 Score", f"{f1_score(y_true, y_pred, average='weighted'):.4f}")
    col5.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

    # ---------------------------
    # Confusion Matrix
    # ---------------------------
    st.subheader("🧩 Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # ---------------------------
    # Classification Report
    # ---------------------------
    st.subheader("📄 Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

else:
    st.info("📥 Upload a **test CSV file** to evaluate the selected model.")
