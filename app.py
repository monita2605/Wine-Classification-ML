import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

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

    # ✅ Automatically detect target column (last column)
    target_column = df.columns[-1]
    st.write(f"Detected Target Column: **{target_column}**")

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Load scaler
    scaler_path = "model/scaler.pkl"

    if not os.path.exists(scaler_path):
        st.error("Scaler file not found. Run train_models.py first.")
        st.stop()

    scaler = pickle.load(open(scaler_path, "rb"))
    X_scaled = scaler.transform(X)

    # Load selected model
    model_path = f"model/{selected_model}.pkl"

    if not os.path.exists(model_path):
        st.error(f"{selected_model}.pkl not found. Train models first.")
        st.stop()

    model = pickle.load(open(model_path, "rb"))

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)

    n_classes = len(np.unique(y))

    if n_classes == 2:
        auc = roc_auc_score(y, y_prob[:, 1])
    else:
        auc = roc_auc_score(y, y_prob, multi_class="ovr")

    accuracy = accuracy_score(y, y_pred)
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
