"""Download and split the UCI Sensorless Drive Diagnosis dataset.

Usage:
    python prepare_data.py
    python prepare_data.py --test-size 0.25 --random-state 31415
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_URL = "https://archive.ics.uci.edu/static/public/325/dataset+for+sensorless+drive+diagnosis.zip"
TARGET_COLUMN = "Class"
FEATURE_COLUMNS = [f"Feature{index:02d}" for index in range(1, 49)]
CLASS_LABELS = list(range(1, 12))
EXPECTED_FEATURE_COUNT = 48
EXPECTED_CLASS_COUNT = 11
EXPECTED_ROW_COUNT = 58509
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 20261106
EXIT_SUCCESS = 0
EXIT_FAILURE = 1


def load_dataset() -> pd.DataFrame:
    """Read 48 unnamed source features and the integer class code from UCI."""
    with urllib.request.urlopen(DATASET_URL, timeout=120) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        filename = "Sensorless_drive_diagnosis.txt"
        if filename not in archive.namelist():
            raise ValueError("The UCI archive is missing the Sensorless Drive data.")
        dataset = pd.read_csv(
            io.BytesIO(archive.read(filename)), sep=r"\s+", header=None
        )
    if dataset.shape[1] != EXPECTED_FEATURE_COUNT + 1:
        raise ValueError("The source must contain 48 numeric features and one target.")
    dataset.columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    validate_dataset(dataset)
    dataset[TARGET_COLUMN] = dataset[TARGET_COLUMN].astype(int)
    return dataset


def validate_dataset(dataset: pd.DataFrame) -> list[str]:
    """Validate source dimensions, finite numeric features, and integer labels."""
    if list(dataset.columns) != [*FEATURE_COLUMNS, TARGET_COLUMN]:
        raise ValueError("Expected Feature01 through Feature48 followed by Class.")
    if len(dataset) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ROW_COUNT} instances; found {len(dataset)}."
        )
    if dataset.isna().any().any():
        raise ValueError("The downloaded dataset contains missing values.")
    if not all(
        pd.api.types.is_numeric_dtype(dataset[column]) for column in FEATURE_COLUMNS
    ):
        raise ValueError("All 48 source features must be numeric.")
    if not np.isfinite(dataset[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("The downloaded dataset contains non-finite feature values.")
    if set(dataset[TARGET_COLUMN].unique()) != set(CLASS_LABELS):
        raise ValueError("Expected integer condition classes 1 through 11.")
    return FEATURE_COLUMNS.copy()


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Download and create stratified train/test CSV files."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser


def main() -> int:
    """Download, validate, split, and save the full dataset."""
    args = create_parser().parse_args()
    if not 0 < args.test_size < 1:
        print("Error: --test-size must be between 0 and 1.", file=sys.stderr)
        return EXIT_FAILURE
    try:
        dataset = load_dataset()
        train_data, test_data = train_test_split(
            dataset,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=dataset[TARGET_COLUMN],
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        train_path = args.output_dir / "train_data.csv"
        test_path = args.output_dir / "test_data.csv"
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        print(f"Dataset: {len(dataset):,} rows, {EXPECTED_FEATURE_COUNT} features")
        print(f"Target: {TARGET_COLUMN} ({EXPECTED_CLASS_COUNT} classes)")
        print(f"Training rows: {len(train_data):,} -> {train_path}")
        print(f"Test rows: {len(test_data):,} -> {test_path}")
        print("Class counts in test data:")
        print(test_data[TARGET_COLUMN].value_counts().sort_index().to_string())
        return EXIT_SUCCESS
    except (
        OSError,
        ValueError,
        RuntimeError,
        urllib.error.URLError,
        zipfile.BadZipFile,
    ) as error:
        print(f"Error preparing dataset: {error}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
