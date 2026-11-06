"""Train and evaluate the assignment's five classifiers.

Usage:
    python train_models.py
    python train_models.py --train-data train_data.csv --test-data test_data.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

TARGET_COLUMN = "Class"
RANDOM_STATE = 20261106
EXIT_SUCCESS = 0
EXIT_FAILURE = 1


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train five classifiers and evaluate them on held-out data."
    )
    parser.add_argument("--train-data", type=Path, default=Path("train_data.csv"))
    parser.add_argument("--test-data", type=Path, default=Path("test_data.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    parser.add_argument("--metrics-file", type=Path, default=Path("model_metrics.csv"))
    return parser


def validate_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> list[str]:
    """Ensure training and test files have compatible numeric features and classes."""
    for label, data in (("training", train_data), ("test", test_data)):
        if TARGET_COLUMN not in data.columns:
            raise ValueError(f"The {label} CSV is missing the {TARGET_COLUMN!r} target.")
        if data.isna().any().any():
            raise ValueError(f"The {label} CSV contains missing values.")

    feature_columns = [column for column in train_data.columns if column != TARGET_COLUMN]
    test_features = [column for column in test_data.columns if column != TARGET_COLUMN]
    if not feature_columns or feature_columns != test_features:
        raise ValueError("Training and test CSVs must have the same feature columns in order.")
    if any(not pd.api.types.is_numeric_dtype(train_data[column]) for column in feature_columns):
        raise ValueError("All model input features must be numeric.")
    if set(train_data[TARGET_COLUMN].unique()) != set(test_data[TARGET_COLUMN].unique()):
        raise ValueError("Training and test CSVs must contain the same target classes.")
    return feature_columns


def build_models() -> dict[str, tuple[str, Pipeline]]:
    """Create all five assignment classifiers with preprocessing where needed."""
    return {
        "logistic_regression": (
            "Logistic Regression",
            make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
            ),
        ),
        "decision_tree": (
            "Decision Tree",
            make_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ),
        "knn": (
            "KNN",
            make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        ),
        "gaussian_naive_bayes": (
            "Naive Bayes (Gaussian)",
            make_pipeline(GaussianNB()),
        ),
        "support_vector_machine": (
            "Support Vector Machine",
            make_pipeline(
                StandardScaler(),
                CalibratedClassifierCV(
                    estimator=SVC(random_state=RANDOM_STATE),
                    method="sigmoid",
                    cv=5,
                ),
            ),
        ),
    }


def evaluate_model(
    model: Pipeline,
    features: pd.DataFrame,
    labels: pd.Series,
) -> dict[str, float]:
    """Calculate the required metrics using macro averaging for multiclass scores."""
    predictions = model.predict(features)
    classes = model.classes_
    auc_scores = model.predict_proba(features)
    binary_labels = label_binarize(labels, classes=classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "Accuracy": float(accuracy_score(labels, predictions)),
        "AUC": float(
            roc_auc_score(
                binary_labels,
                auc_scores,
                average="macro",
            )
        ),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "MCC": float(matthews_corrcoef(labels, predictions)),
    }


def main() -> int:
    """Fit models on training data, evaluate on test data, and save artifacts."""
    args = create_parser().parse_args()
    try:
        train_data = pd.read_csv(args.train_data)
        test_data = pd.read_csv(args.test_data)
        feature_columns = validate_data(train_data, test_data)
        train_features = train_data[feature_columns]
        train_labels = train_data[TARGET_COLUMN]
        test_features = test_data[feature_columns]
        test_labels = test_data[TARGET_COLUMN]

        args.model_dir.mkdir(parents=True, exist_ok=True)
        result_rows: list[dict[str, str | float]] = []
        for model_key, (model_name, model) in build_models().items():
            model.fit(train_features, train_labels)
            metrics = evaluate_model(model, test_features, test_labels)
            model_path = args.model_dir / f"{model_key}.joblib"
            joblib.dump(model, model_path)
            result_rows.append({"ML Model Name": model_name, **metrics})
            print(f"{model_name}: {metrics}")
            print(f"Saved: {model_path}")

        args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result_rows).to_csv(args.metrics_file, index=False)
        print(f"Metrics comparison saved: {args.metrics_file}")
        return EXIT_SUCCESS
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error training or evaluating models: {error}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())