import streamlit as st
import pickle
import numpy as np

st.title("🍷 Wine Classification App")

model_name = st.selectbox(
    "Choose Model",
    ["logistic_regression", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
)

with open(f"model/{model_name}.pkl", "rb") as f:
    model = pickle.load(f)

features = st.text_input("Enter feature values separated by commas")

if st.button("Predict"):
    values = np.array([float(x) for x in features.split(",")]).reshape(1, -1)
    prediction = model.predict(values)
    st.success(f"Prediction: {prediction[0]}")
