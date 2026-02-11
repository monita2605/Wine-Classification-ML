import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Wine Classification ML App", layout="centered")
st.title("🍷 Wine Classification using Machine Learning")

# ---------------------------
# Load models & scaler
# ---------------------------
models = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "Random Forest": "model/random_forest.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "XGBoost": "model/xgboost.pkl"
}

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------------------
# Model selection (Requirement b)
# ---------------------------
model_name = st.selectbox("Select a Model", list(models.keys()))

with open(models[model_name], "rb") as f:
    model = pickle.load(f)

# ---------------------------
# Dataset upload (Requirement a)
# ---------------------------
st.subheader("📂 Upload Test Dataset (CSV only)")

uploaded_file = st.file_uploader(
    "Upload test CSV file (must contain target column)",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("### Preview of Uploaded Data")
    st.dataframe(data.head())

    if "target" not in data.columns:
        st.error("❌ CSV must contain a 'target' column")
    else:
        X = data.drop("target", axis=1)
        y = data["target"]

        # Scaling
        X_scaled = scaler.transform(X)

        # Prediction
        y_pred = model.predict(X_scaled)

        # ---------------------------
        # Evaluation Metrics (Requirement c)
        # ---------------------------
        st.subheader("📊 Evaluation Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Accuracy", round(accuracy_score(y, y_pred), 3))
            st.metric("Precision", round(precision_score(y, y_pred, average="weighted"), 3))

        with col2:
            st.metric("Recall", round(recall_score(y, y_pred, average="weighted"), 3))
            st.metric("F1 Score", round(f1_score(y, y_pred, average="weighted"), 3))

        # ---------------------------
        # Confusion Matrix (Requirement d)
        # ---------------------------
        st.subheader("🔢 Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ---------------------------
        # Classification Report (Requirement d)
        # ---------------------------
        st.subheader("📄 Classification Report")

        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
