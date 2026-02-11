import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# =====================================================
# Create model directory if not exists
# =====================================================
os.makedirs("model", exist_ok=True)

# =====================================================
# Load Dataset
# =====================================================
data = pd.read_csv("wine.csv")

# Separate features & target
X = data.drop("target", axis=1)
y = data["target"]

# Keep numeric columns only
X = X.select_dtypes(include=["number"])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# Models Dictionary
# =====================================================
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(random_state=42),
    "xgboost": XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
}

# =====================================================
# Training Loop
# =====================================================
metrics_list = []

for name, model in models.items():

    print(f"Training {name}...")

    # Create pipeline
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    metrics_list.append(metrics)

    # Save model
    with open(f"model/{name}.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print(f"✅ {name} saved.")

# =====================================================
# Save Metrics CSV
# =====================================================
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv("model/model_metrics.csv", index=False)

print("\n🔥 All models trained and saved successfully!")

print("📊 model_metrics.csv generated.")
