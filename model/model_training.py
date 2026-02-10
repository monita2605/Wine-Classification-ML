import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ---------------------------
# Load dataset
# ---------------------------
data = pd.read_csv("data/wine_train.csv")

X = data.drop("target", axis=1)
y = data["target"]

# ---------------------------
# Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------
# Models (PIPELINE-BASED)
# ---------------------------
models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "decision_tree": Pipeline([
        ("model", DecisionTreeClassifier(random_state=42))
    ]),

    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "naive_bayes": Pipeline([
        ("model", GaussianNB())
    ]),

    "random_forest": Pipeline([
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ]),

    "xgboost": Pipeline([
        ("model", XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        ))
    ])
}

# ---------------------------
# Create model directory
# ---------------------------
os.makedirs("model", exist_ok=True)

metrics = []

# ---------------------------
# Train, evaluate, save
# ---------------------------
for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Handle models without predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = None

    metrics.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1": f1_score(y_test, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

    # 🔥 Save pipeline (scaler + model together)
    with open(f"model/{name}.pkl", "wb") as f:
        pickle.dump(model, f)

# ---------------------------
# Save metrics
# ---------------------------
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("model/model_metrics.csv", index=False)

print("✅ All pipeline-based models trained and saved successfully!")