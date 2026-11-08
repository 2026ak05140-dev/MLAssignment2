"""Streamlit interface for inspecting and evaluating Sensorless Drive classifiers."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from prepare_data import CLASS_LABELS, FEATURE_COLUMNS, TARGET_COLUMN
from train_models import evaluate_model_outputs

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"
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


@st.cache_data(max_entries=2)
def evaluate_saved_models(
    evaluation_data: pd.DataFrame, model_versions: tuple[tuple[str, int], ...]
) -> tuple[dict[str, dict[str, float]], dict[str, pd.Series]]:
    """Cache full comparisons so selecting a focus model does not repeat inference."""
    features = evaluation_data[FEATURE_COLUMNS]
    labels = evaluation_data[TARGET_COLUMN]
    results: dict[str, dict[str, float]] = {}
    predictions: dict[str, pd.Series] = {}
    for model_name, modified_time in model_versions:
        model = load_model(MODEL_DIR / MODEL_FILES[model_name], modified_time)
        if list(model.feature_names_in_) != FEATURE_COLUMNS:
            raise ValueError(
                "A saved model does not match the 48 Sensorless Drive features."
            )
        if set(model.classes_) != set(CLASS_LABELS):
            raise ValueError(
                "A saved model does not match condition classes 1 through 11."
            )
        results[model_name], model_predictions = evaluate_model_outputs(
            model, features, labels
        )
        predictions[model_name] = pd.Series(
            model_predictions, index=evaluation_data.index
        )
    return results, predictions


def main() -> None:
    """Render the interactive classifier evaluation app."""
    st.set_page_config(
        page_title="Drive Diagnosis Classifier Bench",
        page_icon="DD",
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
        @media (max-width: 700px) {
            [data-testid="stAppViewContainer"] .main .block-container {
                padding: 1.25rem 1rem 2rem;
            }
            h1 { font-size: 1.8rem; }
            [data-testid="stMetric"] { padding: 10px 12px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Drive Diagnosis Classifier Bench")
    st.caption(
        "48 motor-current features | 11 condition classes | UCI Sensorless Drive Diagnosis"
    )

    try:
        default_data = load_default_test_data()
    except (OSError, pd.errors.ParserError) as error:
        st.error(f"Could not load the bundled test data: {error}")
        st.stop()

    feature_columns = FEATURE_COLUMNS
    with st.sidebar:
        st.header("Evaluation")
        selected_model_name = st.selectbox("Focus model", list(MODEL_FILES))
        uploaded_file = st.file_uploader("Upload test CSV", type="csv")
        st.caption(
            "Upload test data only: Feature01 through Feature48 and the integer Class target."
        )
        st.download_button(
            "Download example test CSV",
            data=default_data.to_csv(index=False).encode("utf-8"),
            file_name="test_data.csv",
            mime="text/csv",
            width="stretch",
        )

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
    missing_columns = [
        column for column in required_columns if column not in evaluation_data
    ]
    if missing_columns:
        st.error(f"CSV is missing required columns: {', '.join(missing_columns)}")
        st.stop()
    if evaluation_data[required_columns].isna().any().any():
        st.error("The test CSV contains missing values in required columns.")
        st.stop()
    if not all(
        pd.api.types.is_numeric_dtype(evaluation_data[column])
        for column in feature_columns
    ):
        st.error("All 48 input features must be numeric.")
        st.stop()
    if not np.isfinite(evaluation_data[feature_columns].to_numpy()).all():
        st.error("The test CSV contains non-finite feature values.")
        st.stop()

    features = evaluation_data[feature_columns]
    labels = evaluation_data[TARGET_COLUMN]
    if labels.nunique() < 2:
        st.error(
            "The target column must contain at least two classes to calculate AUC."
        )
        st.stop()

    if set(labels.unique()) != set(CLASS_LABELS):
        st.error(
            "The test CSV must contain all integer condition classes 1 through 11 and no others."
        )
        st.stop()

    try:
        model_versions = tuple(
            (name, (MODEL_DIR / filename).stat().st_mtime_ns)
            for name, filename in MODEL_FILES.items()
        )
        with st.spinner("Evaluating all five saved models on test data..."):
            results, predictions_by_model = evaluate_saved_models(
                evaluation_data, model_versions
            )
    except (OSError, ValueError, AttributeError) as error:
        st.error(f"Could not evaluate the saved models: {error}")
        st.stop()

    st.write(f"**Data:** {data_source} · **Rows:** {len(evaluation_data):,}")
    with st.expander("Dataset sample, input schema, and class distribution"):
        st.write(
            "Source: UCI Sensorless Drive Diagnosis, 58,509 rows and 48 numeric features. "
            "The bundled split has 46,807 training rows and 11,702 test rows (seed 20261106). "
            "The final source column is Class; it is excluded from predictors. "
            "Feature01 through Feature48 are positional names assigned to unnamed source inputs."
        )
        st.dataframe(
            evaluation_data[[TARGET_COLUMN, *feature_columns]].head(10),
            width="stretch",
        )
        st.code(
            fill(", ".join([*FEATURE_COLUMNS, TARGET_COLUMN]), width=100),
            language="text",
        )
        distribution = pd.DataFrame(
            {
                "Bundled test rows": default_data[TARGET_COLUMN].value_counts(),
                "Current evaluation rows": labels.value_counts(),
            }
        ).reindex(CLASS_LABELS)
        distribution.index.name = "Condition class"
        st.dataframe(
            distribution, width="stretch", height=35 * (len(CLASS_LABELS) + 1) + 3
        )
        st.caption(
            "Class codes 1 through 11 are preserved; specific fault-name mappings are not supplied."
        )
    st.subheader("Model comparison")
    comparison = pd.DataFrame.from_dict(results, orient="index")
    comparison.index.name = "ML Model Name"
    st.dataframe(
        comparison.style.format("{:.4f}"),
        width="stretch",
    )
    st.download_button(
        "Download model comparison",
        data=comparison.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
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
    selected_model = load_model(
        selected_model_path, selected_model_path.stat().st_mtime_ns
    )
    selected_predictions = predictions_by_model[selected_model_name]
    predictions_frame = pd.DataFrame(
        {
            "Record": np.arange(1, len(evaluation_data) + 1),
            "Actual class": labels.to_numpy(),
            "Predicted class": selected_predictions.to_numpy(),
        }
    )
    predictions_frame["Correct"] = predictions_frame["Actual class"].eq(
        predictions_frame["Predicted class"]
    )
    st.download_button(
        "Download selected-model predictions",
        data=predictions_frame.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_model_name.lower().replace(' ', '_')}_predictions.csv",
        mime="text/csv",
    )
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
        st.dataframe(
            matrix_frame, width="stretch", height=35 * (len(class_labels) + 1) + 3
        )
    with report_column:
        st.subheader("Classification report")
        report = classification_report(
            labels,
            selected_predictions,
            labels=class_labels,
            zero_division=0,
            output_dict=True,
        )
        st.dataframe(
            pd.DataFrame(report).transpose().round(4),
            width="stretch",
            height=35 * (len(class_labels) + 4) + 3,
        )

    with st.expander("Trained pipeline and prediction examples"):
        st.write(
            "This is the saved fitted pipeline, not a model trained on your upload. "
            "StandardScaler and SVM calibration are fitted using training data only."
        )
        st.code(str(selected_model), language="python")
        examples = (
            evaluation_data[["Feature01", "Feature02", TARGET_COLUMN]].head(10).copy()
        )
        examples = examples.rename(columns={TARGET_COLUMN: "Actual class"})
        examples["Predicted class"] = selected_predictions.head(10)
        examples["Correct"] = examples["Actual class"].eq(examples["Predicted class"])
        examples["Highest class probability"] = selected_model.predict_proba(
            features.head(10)
        ).max(axis=1)
        st.dataframe(examples, width="stretch")
        st.caption("Probabilities are model estimates, not guarantees of correctness.")


if __name__ == "__main__":
    main()
