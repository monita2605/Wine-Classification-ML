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

# Expected features used during training
expected_features = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Detect target column automatically (last column)
    target_column = df.columns[-1]
    st.write(f"Detected Target Column: **{target_column}**")

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # ⭐ FIX 1 — Convert wine quality to classification labels
    # (IMPORTANT for ROC-AUC & classification models)
    if y.dtype != "object":
        if y.nunique() > 3:
            y = (y >= 7).astype(int)
            st.info("Converted quality scores → classification labels (Good=1, Bad=0)")

    # ⭐ FIX 2 — Ensure test features match training features
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0

    X = X[expected_features]  # reorder columns correctly

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

    # Predictions
    y_pred = model.predict(X_scaled)

    # Some models may not support predict_proba
    try:
        y_prob = model.predict_proba(X_scaled)
        prob_available = True
    except:
        prob_available = False

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="weighted")
    recall = recall_score(y, y_pred, average="weighted")
    f1 = f1_score(y, y_pred, average="weighted")
    mcc = matthews_corrcoef(y, y_pred)

    # ⭐ FIX 3 — Safe ROC-AUC calculation
    try:
        if prob_available:
            n_classes = len(np.unique(y))
            if n_classes == 2:
                auc = roc_auc_score(y, y_prob[:, 1])
            else:
                auc = roc_auc_score(y, y_prob, multi_class="ovr")
        else:
            auc = 0
    except:
        auc = 0

    # Display Metrics
    st.subheader("📊 Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"AUC Score: {auc:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"MCC Score: {mcc:.4f}")

    # Confusion Matrix
    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.success("App ran successfully 🎉 Upload another dataset to test again.")
