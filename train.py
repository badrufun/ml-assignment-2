import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import MODEL_DIR, MODEL_FILES, PROJECT_DIR, TARGET, evaluate

DATA_URL = "https://archive.ics.uci.edu/static/public/198/data.csv"
FAULT_COLUMNS = [
    "Pastry", "Z_Scratch", "K_Scratch", "Stains",
    "Dirtiness", "Bumps", "Other_Faults",
]
SEED = 42


def main():
    data = pd.read_csv(DATA_URL)
    indicators = data[FAULT_COLUMNS]
    if not indicators.isin([0, 1]).all().all() or not indicators.sum(axis=1).eq(1).all():
        raise ValueError("Each source row must have exactly one fault label.")

    features = data.drop(columns=FAULT_COLUMNS)
    target = indicators.idxmax(axis=1).rename(TARGET)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=SEED,
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "Gaussian Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "Support Vector Machine": make_pipeline(
            StandardScaler(), SVC(probability=True, random_state=SEED)
        ),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        joblib.dump(model, MODEL_DIR / MODEL_FILES[name], compress=3)
        results[name], _ = evaluate(model, x_test, y_test)

    x_test.assign(Fault=y_test).to_csv(PROJECT_DIR / "test_data.csv", index=False)
    print(f"Training rows: {len(x_train)} | Test rows: {len(x_test)}")
    print(pd.DataFrame.from_dict(results, orient="index").round(4).to_string())
    print("\nTest class counts:")
    print(y_test.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
