from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
TARGET = "Fault"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
    "Support Vector Machine": "svm.joblib",
}


def evaluate(model, features, actual):
    predicted = model.predict(features)
    labels = model.classes_
    auc = None
    if set(actual) == set(labels):
        auc = roc_auc_score(
            actual, model.predict_proba(features), labels=labels,
            multi_class="ovr", average="macro",
        )

    options = {"labels": labels, "average": "macro", "zero_division": 0}
    metrics = {
        "Accuracy": accuracy_score(actual, predicted),
        "AUC": auc,
        "Precision": precision_score(actual, predicted, **options),
        "Recall": recall_score(actual, predicted, **options),
        "F1": f1_score(actual, predicted, **options),
        "MCC": matthews_corrcoef(actual, predicted),
    }
    return metrics, predicted
