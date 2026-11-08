import altair as alt
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

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


@st.cache_data
def compare_models():
    reference_model = load_model(next(iter(MODEL_FILES)))
    features, actual = read_test_data(PROJECT_DIR / "test_data.csv", reference_model)
    results = {}
    for name in MODEL_FILES:
        results[name], _ = evaluate(load_model(name), features, actual)
    return pd.DataFrame(results).T.rename_axis("Model").round(4)


def show_confusion_matrix(matrix):
    cells = matrix.rename_axis(index="Actual", columns="Predicted").stack().rename("Count").reset_index()
    base = alt.Chart(cells).encode(
        x=alt.X("Predicted:N", axis=alt.Axis(labelAngle=-35)),
        y=alt.Y("Actual:N"),
        tooltip=["Actual:N", "Predicted:N", "Count:Q"],
    )
    boxes = base.mark_rect().encode(color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")))
    numbers = base.mark_text(fontSize=13).encode(
        text="Count:Q",
        color=alt.condition(
            alt.datum.Count > cells["Count"].max() / 2,
            alt.value("white"), alt.value("#172b40"),
        ),
    )
    st.altair_chart((boxes + numbers).properties(height=340), width=620)


def main():
    st.set_page_config(page_title="Steel Plate Fault Classification", layout="wide")
    st.title("Steel Plate Fault Classification")
    st.write("Compare five classifiers for seven types of steel plate faults.")

    st.subheader("Model Performance Comparison")
    with st.expander("Compare all five models", expanded=True):
        st.caption("These scores use the included test set for a fair comparison.")
        st.dataframe(compare_models(), width="stretch")

    st.subheader("1. Test Dataset")
    upload = st.file_uploader("Test CSV (optional)", type="csv")
    source = upload
    if upload is None:
        source = PROJECT_DIR / "test_data.csv"
        st.caption("Using the included test_data.csv. Upload a labeled CSV to use it instead.")

    preview = st.container()
    st.subheader("3. Select Classification Model")
    name = st.selectbox("Choose a trained model", MODEL_FILES)
    model = load_model(name)
    try:
        features, actual = read_test_data(source, model)
    except ValueError as exc:
        st.error(str(exc))
        return

    with preview:
        st.subheader("2. Dataset Preview")
        rows, inputs, target = st.columns(3)
        rows.metric("Test rows", len(actual))
        inputs.metric("Input features", features.shape[1])
        target.metric("Target column", TARGET)
        with st.expander("View the first 10 rows"):
            st.dataframe(features.head(10).assign(Fault=actual.head(10)), width="stretch")

    metrics, predicted = evaluate(model, features, actual)
    st.subheader("4. Model Evaluation Results")
    metric_items = list(metrics.items())
    for start in (0, 3):
        for column, (label, value) in zip(st.columns(3), metric_items[start:start + 3]):
            column.metric(label, f"{value:.4f}" if value is not None else "N/A")
    if metrics["AUC"] is None:
        st.caption("AUC requires at least one test row from each fault class.")

    st.subheader("5. Confusion Matrix")
    labels = model.classes_
    matrix = confusion_matrix(actual, predicted, labels=labels)
    show_confusion_matrix(pd.DataFrame(matrix, index=labels, columns=labels))
    st.caption("Rows show actual faults and columns show predicted faults. The diagonal shows correct predictions.")

    st.subheader("6. Classification Report")
    report = classification_report(
        actual, predicted, labels=labels, output_dict=True, zero_division=0,
    )
    scores = pd.DataFrame(report).T.loc[[*labels, "macro avg", "weighted avg"]].round(4)
    scores["support"] = scores["support"].astype(int)
    st.dataframe(scores)


if __name__ == "__main__":
    main()
