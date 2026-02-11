import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ---------------------------
# Load dataset (ALL 13 FEATURES)
# ---------------------------
data = pd.read_csv("data/wine_train.csv")

# Ensure we are using ALL columns except target
X = data.drop("target", axis=1)
y = data["target"]

print("Number of features used for training:", X.shape[1])

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
# Define Pipeline-based models
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
        ("model", KNeighborsClassifier())
    ]),

    "naive_bayes": Pipeline([
        ("model", GaussianNB())
    ]),

    "random_forest": Pipeline([
        ("model", RandomForestClassifier(random_state=42))
    ]),

    "xgboost": Pipeline([
        ("model", XGBClassifier(
            eval_metric="mlogloss",
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
for name, pipeline in models.items():

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    else:
        auc = None

    metrics.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1": f1_score(y_test, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

    with open(f"model/{name}.pkl", "wb") as f:
        pickle.dump(pipeline, f)

# Save metrics
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("model/model_metrics.csv", index=False)

print("✅ Training completed successfully using ALL 13 features.")