import streamlit as st
import pandas as pd
import numpy as np
import pickle

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

# =====================================================
# App Configuration
# =====================================================
st.set_page_config(page_title="Wine Classification", layout="wide")
st.title("🍷 Wine Classification – ML Model Evaluation")

# =====================================================
# Model Paths
# =====================================================
MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

# =====================================================
# Sidebar
# =====================================================
st.sidebar.header("⚙️ Controls")

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_PATHS.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset (CSV)",
    type=["csv"]
)

# =====================================================
# Load Model (Cached)
# =====================================================
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model(MODEL_PATHS[selected_model])
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# =====================================================
# Get Expected Features
# =====================================================
if hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)
else:
    expected_features = None

# =====================================================
# Show Model Comparison Table
# =====================================================
try:
    metrics_df = pd.read_csv("model/model_metrics.csv")
    st.subheader("📊 Model Comparison Table")
    st.dataframe(metrics_df, use_container_width=True)
except:
    st.warning("⚠️ model_metrics.csv not found in model folder.")

# =====================================================
# Dataset Processing
# =====================================================
if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Unable to read CSV: {e}")
        st.stop()

    st.subheader("📂 Uploaded Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    try:
        # -------------------------------------------------
        # Separate Target (Optional)
        # -------------------------------------------------
        if "target" in data.columns:
            y_true = data["target"]
            X = data.drop("target", axis=1)
        else:
            y_true = None
            X = data.copy()

        # -------------------------------------------------
        # Align Features
        # -------------------------------------------------
        if expected_features is not None:

            # Add missing columns as 0
            for col in expected_features:
                if col not in X.columns:
                    X[col] = 0

            # Remove extra columns
            X = X[expected_features]

        # -------------------------------------------------
        # Convert to Numeric
        # -------------------------------------------------
        X = X.apply(pd.to_numeric, errors="coerce")

        # -------------------------------------------------
        # Fill Missing Values with Column Means
        # -------------------------------------------------
        if X.isnull().sum().sum() > 0:
            st.warning("⚠️ Missing/non-numeric values detected. Filling with column means.")
            X = X.fillna(X.mean())

        # Final NaN check
        if X.isnull().sum().sum() > 0:
            st.error("❌ Dataset still contains invalid values after cleaning.")
            st.stop()

        # Final shape safety check
        if expected_features is not None:
            if X.shape[1] != len(expected_features):
                st.error("❌ Feature count mismatch.")
                st.stop()

        # -------------------------------------------------
        # Make Predictions
        # -------------------------------------------------
        y_pred = model.predict(X)

        # =================================================
        # If Ground Truth Exists → Show Metrics
        # =================================================
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

        # =================================================
        # If No Target → Show Predictions + Download
        # =================================================
        else:
            data["Predicted_Class"] = y_pred

            st.subheader("🔮 Prediction Results")
            st.dataframe(data, use_container_width=True)

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download Predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

else:
    st.info("📥 Upload a CSV dataset to begin.")
