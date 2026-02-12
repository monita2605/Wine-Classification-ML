import os
import pickle
import pandas as pd
import numpy as np

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

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False


def load_data():
    train_df = pd.read_csv("wine_train.csv")
    test_df = pd.read_csv("wine_test.csv")

    # Automatically detect target column (last column)
    target_column = train_df.columns[-1]
    print(f"Detected target column: {target_column}")

    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]

    X_test = test_df.drop(target_column, axis=1)
    y_test = test_df[target_column]

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("model", exist_ok=True)
    pickle.dump(scaler, open("model/scaler.pkl", "wb"))

    return X_train_scaled, X_test_scaled


def get_models():
    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "knn": KNeighborsClassifier(),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(random_state=42),
    }

    if xgb_available:
        models["xgboost"] = XGBClassifier(eval_metric="logloss")

    return models


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    n_classes = len(np.unique(y_test))

    if n_classes == 2:
        auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr")

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print("Scaling features...")
    X_train, X_test = scale_data(X_train, X_test)

    models = get_models()
    results = []

    print("\nTraining models and generating pickle files...\n")

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        pickle.dump(model, open(f"model/{name}.pkl", "wb"))

        results.append([name] + list(metrics.values()))
        print(f"✅ {name}.pkl saved")

    results_df = pd.DataFrame(
        results,
        columns=["Model", "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
    )

    print("\nModel Performance Comparison:\n")
    print(results_df)


if __name__ == "__main__":
    main()
