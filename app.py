import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

from model.logistic_regression import train_lr
from model.decision_tree import train_dt
from model.knn import train_knn
from model.naive_bayes import train_nb
from model.random_forest import train_rf
from model.xgboost_model import train_xgb

# Create directory for saved models
os.makedirs("model/saved_models", exist_ok=True)

# Load dataset
train_df = pd.read_csv("wine_train.csv")
test_df = pd.read_csv("wine_test.csv")

X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

models = {
    "logistic_regression": train_lr,
    "decision_tree": train_dt,
    "knn": train_knn,
    "naive_bayes": train_nb,
    "random_forest": train_rf,
    "xgboost": train_xgb
}

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

for name, train_func in models.items():
    model = train_func(X_train, y_train)

    # Save model as .pkl
    model_path = f"model/saved_models/{name}.pkl"
    joblib.dump(model, model_path)

    results = evaluate(model, X_test, y_test)

    print(f"\n{name.upper()}")
    print(f"Model saved at: {model_path}")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")