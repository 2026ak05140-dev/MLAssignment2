"""Streamlit interface for evaluating the saved Dry Bean classifiers."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"
TARGET_COLUMN = "Class"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes (Gaussian)": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


@st.cache_data
def load_default_test_data() -> pd.DataFrame:
    """Load the repository's held-out test data."""
    return pd.read_csv(TEST_DATA_PATH)


@st.cache_resource
def load_model(model_path: Path, modified_time: int):
    """Load a trained model pipeline from disk."""
    return joblib.load(model_path)


def calculate_metrics(model, features: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    """Calculate macro-averaged multiclass metrics for a fitted model."""
    predictions = model.predict(features)
    auc_scores = model.predict_proba(features)
    binary_labels = label_binarize(labels, classes=model.classes_)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "Accuracy": float(accuracy_score(labels, predictions)),
        "AUC": float(roc_auc_score(binary_labels, auc_scores, average="macro")),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "MCC": float(matthews_corrcoef(labels, predictions)),
    }


def main() -> None:
    """Render the interactive classifier evaluation app."""
    st.set_page_config(
        page_title="Dry Bean Classifier Bench",
        page_icon="DB",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        :root {
            --ink: #172a27;
            --muted: #60716d;
            --accent: #147d69;
            --accent-soft: #e4f2ed;
            --rule: #dce5e1;
        }
        [data-testid="stAppViewContainer"] { background: #f7f9f7; color: var(--ink); }
        [data-testid="stSidebar"] { background: #eef4f1; border-right: 1px solid var(--rule); }
        h1, h2, h3 { color: var(--ink); font-family: Georgia, serif; }
        [data-testid="stMetric"] {
            background: white; border: 1px solid var(--rule); border-top: 3px solid var(--accent);
            padding: 14px 16px; border-radius: 6px;
        }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        div[data-testid="stDataFrame"] { border: 1px solid var(--rule); border-radius: 6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Dry Bean Classifier Bench")
    st.caption("Seven-class evaluation on the held-out UCI Dry Bean test split")

    try:
        default_data = load_default_test_data()
    except (OSError, pd.errors.ParserError) as error:
        st.error(f"Could not load the bundled test data: {error}")
        st.stop()

    feature_columns = [column for column in default_data.columns if column != TARGET_COLUMN]
    with st.sidebar:
        st.header("Evaluation")
        selected_model_name = st.selectbox("Focus model", list(MODEL_FILES))
        uploaded_file = st.file_uploader("Upload test CSV", type="csv")
        st.caption("The CSV must include the 16 features and the Class target column.")

    if uploaded_file is None:
        evaluation_data = default_data
        data_source = "Bundled test data"
    else:
        try:
            evaluation_data = pd.read_csv(uploaded_file)
            data_source = uploaded_file.name
        except (UnicodeDecodeError, pd.errors.ParserError) as error:
            st.error(f"Could not read the uploaded CSV: {error}")
            st.stop()

    required_columns = [*feature_columns, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in evaluation_data]
    if missing_columns:
        st.error(f"CSV is missing required columns: {', '.join(missing_columns)}")
        st.stop()
    if evaluation_data[required_columns].isna().any().any():
        st.error("The test CSV contains missing values in required columns.")
        st.stop()
    if not all(
        pd.api.types.is_numeric_dtype(evaluation_data[column]) for column in feature_columns
    ):
        st.error("All 16 input features must be numeric.")
        st.stop()

    features = evaluation_data[feature_columns]
    labels = evaluation_data[TARGET_COLUMN]
    if labels.nunique() < 2:
        st.error("The target column must contain at least two classes to calculate AUC.")
        st.stop()

    reference_model_path = MODEL_DIR / MODEL_FILES[selected_model_name]
    reference_model = load_model(reference_model_path, reference_model_path.stat().st_mtime_ns)
    if set(labels.unique()) != set(reference_model.classes_):
        st.error("The test CSV must contain all seven Dry Bean target classes and no others.")
        st.stop()

    results: dict[str, dict[str, float]] = {}
    predictions_by_model: dict[str, pd.Series] = {}
    try:
        for model_name, model_filename in MODEL_FILES.items():
            model_path = MODEL_DIR / model_filename
            model = load_model(model_path, model_path.stat().st_mtime_ns)
            missing_features = [
                feature for feature in model.feature_names_in_ if feature not in features.columns
            ]
            if missing_features:
                raise ValueError(f"Model input features are missing: {', '.join(missing_features)}")
            model_features = features[model.feature_names_in_]
            results[model_name] = calculate_metrics(model, model_features, labels)
            predictions_by_model[model_name] = pd.Series(
                model.predict(model_features), index=evaluation_data.index
            )
    except (OSError, ValueError, AttributeError) as error:
        st.error(f"Could not evaluate the saved models: {error}")
        st.stop()

    st.write(f"**Data:** {data_source} · **Rows:** {len(evaluation_data):,}")
    st.subheader("Model comparison")
    comparison = pd.DataFrame.from_dict(results, orient="index")
    comparison.index.name = "ML Model Name"
    st.dataframe(
        comparison.style.format("{:.4f}"),
        width="stretch",
    )

    st.subheader(selected_model_name)
    selected_metrics = results[selected_model_name]
    for metric_start in (0, 3):
        metric_columns = st.columns(3)
        for column, metric_name in zip(
            metric_columns,
            METRIC_COLUMNS[metric_start : metric_start + 3],
        ):
            column.metric(metric_name, f"{selected_metrics[metric_name]:.4f}")

    selected_model_path = MODEL_DIR / MODEL_FILES[selected_model_name]
    selected_model = load_model(selected_model_path, selected_model_path.stat().st_mtime_ns)
    selected_predictions = predictions_by_model[selected_model_name]
    class_labels = list(selected_model.classes_)
    matrix = confusion_matrix(labels, selected_predictions, labels=class_labels)
    matrix_frame = pd.DataFrame(
        matrix,
        index=[f"Actual {label}" for label in class_labels],
        columns=[f"Predicted {label}" for label in class_labels],
    )
    matrix_column, report_column = st.columns([1.2, 1])
    with matrix_column:
        st.subheader("Confusion matrix")
        st.dataframe(matrix_frame, width="stretch")
    with report_column:
        st.subheader("Classification report")
        report = classification_report(
            labels,
            selected_predictions,
            labels=class_labels,
            zero_division=0,
            output_dict=True,
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), width="stretch")


if __name__ == "__main__":
    main()