import streamlit as st
import pandas as pd
import pickle
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Wine Classification App", layout="centered")
st.title("🍷 Wine Classification using ML Models")

MODEL_OPTIONS = [
    "logistic",
    "decision_tree",
    "knn",
    "naive_bayes",
    "random_forest",
    "xgboost"
]

uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])
selected_model = st.selectbox("Select Model", MODEL_OPTIONS)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "target" not in df.columns:
        st.error("Dataset must contain 'target' column.")
        st.stop()

    X = df.drop("target", axis=1)
    y = df["target"]

    # Load scaler
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    X_scaled = scaler.transform(X)

    # Load selected model
    model_path = f"model/{selected_model}.pkl"

    if not os.path.exists(model_path):
        st.error("Model file not found. Please train models first.")
        st.stop()

    model = pickle.load(open(model_path, "rb"))
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob, multi_class="ovr")
    precision = precision_score(y, y_pred, average="weighted")
    recall = recall_score(y, y_pred, average="weighted")
    f1 = f1_score(y, y_pred, average="weighted")
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("📊 Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"AUC Score: {auc:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"MCC Score: {mcc:.4f}")

    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)
