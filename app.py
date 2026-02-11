import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

st.title("Wine Classification - ML Model Comparison")

# Load dataset
train_df = pd.read_csv("wine_train.csv")
test_df = pd.read_csv("wine_test.csv")

X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

# Load saved models
model_dir = "model/saved_models"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]

models = {}
for file in model_files:
    model_name = file.replace(".pkl", "")
    models[model_name] = joblib.load(os.path.join(model_dir, file))

# -----------------------------
# Model Evaluation
# -----------------------------
results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

results_df = pd.DataFrame(results)

st.subheader("📊 Model Comparison Table")
st.dataframe(results_df)

# -----------------------------
# Plot Comparison
# -----------------------------
st.subheader("📈 Model Performance Comparison")

metric = st.selectbox(
    "Select Metric to Compare",
    ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
)

plt.figure()
plt.bar(results_df["Model"], results_df[metric])
plt.xticks(rotation=45)
plt.ylabel(metric)
plt.title(f"{metric} Comparison Across Models")

st.pyplot(plt)

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("🔍 Make a Prediction")

selected_model_name = st.selectbox("Choose Model", list(models.keys()))
selected_model = models[selected_model_name]

input_data = []

for column in X_test.columns:
    value = st.number_input(f"Enter {column}", value=float(X_test[column].mean()))
    input_data.append(value)

if st.button("Predict"):
    prediction = selected_model.predict([input_data])
    st.success(f"Prediction: {prediction[0]}")
