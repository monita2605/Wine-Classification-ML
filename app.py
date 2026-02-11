import streamlit as st
import pandas as pd
import pickle
import numpy as np

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
st.title("🍷 Wine Classification – ML Model Evaluation")

# ---------------------------
# Model Paths
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
# Sidebar
# ---------------------------
st.sidebar.header("⚙️ Controls")

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_PATHS.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset (CSV)",
    type=["csv"]
)

# ---------------------------
# Load Model
# ---------------------------
with open(MODEL_PATHS[selected_model], "rb") as f:
    model = pickle.load(f)

expected_features = list(model.feature_names_in_)

# ---------------------------
# Show Comparison Table
# ---------------------------
try:
    metrics_df = pd.read_csv("model/model_metrics.csv")
    st.subheader("📊 Model Comparison Table")
    st.dataframe(metrics_df, use_container_width=True)
except:
    st.warning("Model metrics file not found.")

# ---------------------------
# Dataset Processing
# ---------------------------
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📂 Uploaded Dataset")
    st.dataframe(data.head(), use_container_width=True)

    try:
        # Separate target if exists
        if "target" in data.columns:
            y_true = data["target"]
            X = data.drop("target", axis=1)
        else:
            y_true = None
            X = data.copy()

        # Align features
        missing_cols = [col for col in expected_features if col not in X.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.stop()

        X = X[expected_features]

        # Convert to numeric
        X = X.apply(pd.to_numeric, errors="coerce")

        if X.isnull().sum().sum() > 0:
            st.error("❌ Dataset contains non-numeric or missing values.")
            st.stop()

        # Prediction
        y_pred = model.predict(X)

        # If ground truth available
        if y_true is not None:

            st.subheader("✅ Model Performance")

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
            col2.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.4f}")
            col3.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.4f}")

            col4, col5 = st.columns(2)
            col4.metric("F1 Score", f"{f1_score(y_true, y_pred, average='weighted'):.4f}")
            col5.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

            # Confusion Matrix
            st.subheader("🧩 Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # Classification Report
            st.subheader("📄 Classification Report")
            report = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

        else:
            data["Predicted_Class"] = y_pred
            st.subheader("🔮 Prediction Results")
            st.dataframe(data, use_container_width=True)

    except Exception as e:
        st.error("❌ Prediction failed. Please ensure dataset matches training features.")

else:
    st.info("📥 Upload a CSV dataset to begin.")