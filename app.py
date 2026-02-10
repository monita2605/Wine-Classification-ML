import streamlit as st
import pickle
import numpy as np
import os

# ---------------------------
# Load trained model
# ---------------------------
with open("model/logistic_regression.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Wine Classification", layout="centered")
st.title("🍷 Wine Classification – Logistic Regression")

st.write("Enter the wine chemical properties below:")

# Feature names (Wine dataset)
feature_names = [
    "Alcohol",
    "Malic Acid",
    "Ash",
    "Alcalinity of Ash",
    "Magnesium",
    "Total Phenols",
    "Flavanoids",
    "Nonflavanoid Phenols",
    "Proanthocyanins",
    "Color Intensity",
    "Hue",
    "OD280/OD315 of Diluted Wines",
    "Proline"
]

inputs = []

# Create number inputs
for feature in feature_names:
    value = st.number_input(feature, value=0.0)
    inputs.append(value)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Wine Class"):
    values = np.array(inputs).reshape(1, -1)

    # Safety check
    if values.shape[1] != model.n_features_in_:
        st.error(
            f"Model expects {model.n_features_in_} features, "
            f"but received {values.shape[1]}"
        )
    else:
        # Scale input
        values_scaled = scaler.transform(values)

        prediction = model.predict(values_scaled)

        st.success(f"🍷 Predicted Wine Class: {prediction[0]}")
