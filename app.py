import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

from evaluation import MODEL_DIR, MODEL_FILES, PROJECT_DIR, TARGET, evaluate


@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


def read_test_data(source, model):
    try:
        data = pd.read_csv(source)
    except (ValueError, pd.errors.ParserError) as exc:
        raise ValueError("Could not read the file as a CSV.") from exc

    columns = list(model.feature_names_in_)
    missing = [column for column in [*columns, TARGET] if column not in data]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if data.empty:
        raise ValueError("The CSV has no test rows.")

    features = data[columns].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any() or features.isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("Every feature must have a finite numeric value.")
    actual = data[TARGET]
    if not actual.isin(model.classes_).all():
        raise ValueError("The Fault column contains a missing or unknown fault label.")

    return features, actual


def main():
    st.set_page_config(page_title="Steel Plate Fault Classification")
    st.title("Steel Plate Fault Classification")
    upload = st.file_uploader("Test CSV (optional)", type="csv")
    name = st.selectbox("Model", MODEL_FILES)
    source = upload
    if upload is None:
        source = PROJECT_DIR / "test_data.csv"
        st.caption("Using the included test_data.csv. Upload a labeled CSV to use it instead.")

    model = load_model(name)
    try:
        features, actual = read_test_data(source, model)
    except ValueError as exc:
        st.error(str(exc))
        return

    metrics, predicted = evaluate(model, features, actual)
    st.subheader("Evaluation metrics")
    st.write(f"Test rows: {len(actual)}")
    st.table(pd.DataFrame({
        "Metric": metrics.keys(),
        "Score": [f"{value:.4f}" if value is not None else "N/A" for value in metrics.values()],
    }).set_index("Metric"))
    if metrics["AUC"] is None:
        st.caption("AUC requires at least one test row from each fault class.")

    st.subheader("Confusion matrix")
    labels = model.classes_
    matrix = confusion_matrix(actual, predicted, labels=labels)
    st.dataframe(pd.DataFrame(matrix, index=labels, columns=labels))
    st.caption("Rows are actual fault types. Columns are predicted fault types.")


if __name__ == "__main__":
    main()
