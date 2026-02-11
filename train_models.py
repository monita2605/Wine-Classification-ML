import os
import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def load_data():
    train_df = pd.read_csv("wine_train.csv")
    test_df = pd.read_csv("wine_test.csv")

    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]

    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("model", exist_ok=True)
    pickle.dump(scaler, open("model/scaler.pkl", "wb"))

    return X_train_scaled, X_test_scaled


def get_models():
    return {
        "logistic": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "knn": KNeighborsClassifier(),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(random_state=42),
        "xgboost": XGBClassifier(eval_metric="mlogloss")
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob, multi_class="ovr"),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }


def main():
    X_train, X_test, y_train, y_test = load_data()
    X_train, X_test = scale_features(X_train, X_test)

    models = get_models()
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        results.append([name] + list(metrics.values()))

        pickle.dump(model, open(f"model/{name}.pkl", "wb"))

    results_df = pd.DataFrame(
        results,
        columns=["Model", "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
    )

    print("\nModel Performance Comparison:\n")
    print(results_df)



