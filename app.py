import streamlit as st
import pickle
import numpy as np

# ---------------------------
# Load model
# ---------------------------
with open("model/logistic_regression.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Wine Classification", layout="centered")
st.title("🍷 Wine Classification – Logistic Regression")

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

for feature in feature_names:
    inputs.append(st.number_input(feature, value=0.0))

if st.button("Predict Wine Class"):
    values = np.array(inputs).reshape(1, -1)

    if values.shape[1] != model.n_features_in_:
        st.error(
            f"Model expects {model.n_features_in_} features, "
            f"but received {values.shape[1]}"
        )
    else:
        prediction = model.predict(values)
        st.success(f"🍷 Predicted Wine Class: {prediction[0]}")
