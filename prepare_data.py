"""Download and split the UCI Dry Bean dataset.

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

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
TARGET_COLUMN = "Class"
EXPECTED_FEATURE_COUNT = 16
EXPECTED_CLASS_COUNT = 7
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 20261106
EXIT_SUCCESS = 0
EXIT_FAILURE = 1


def load_dataset() -> pd.DataFrame:
    """Download the official UCI archive and read its Excel dataset."""
    with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        workbook_paths = [
            name for name in archive.namelist() if name.lower().endswith(".xlsx")
        ]
        if not workbook_paths:
            raise ValueError("The UCI archive does not contain an Excel workbook.")
        workbook = archive.read(workbook_paths[0])
        return pd.read_excel(io.BytesIO(workbook))


def validate_dataset(dataset: pd.DataFrame) -> list[str]:
    """Validate the assignment's row, feature, and target requirements."""
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Expected target column {TARGET_COLUMN!r} was not found.")

    feature_columns = [column for column in dataset.columns if column != TARGET_COLUMN]
    if len(feature_columns) < 12:
        raise ValueError(f"Expected at least 12 features; found {len(feature_columns)}.")
    if len(dataset) < 500:
        raise ValueError(f"Expected at least 500 instances; found {len(dataset)}.")
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} Dry Bean features; found {len(feature_columns)}."
        )
    if dataset[TARGET_COLUMN].nunique() != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CLASS_COUNT} target classes; "
            f"found {dataset[TARGET_COLUMN].nunique()}."
        )
    if dataset.isna().any().any():
        raise ValueError("The downloaded dataset contains missing values.")
    return feature_columns


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Download and create stratified train/test CSV files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for train_data.csv and test_data.csv (default: current directory).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of rows reserved for testing (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Seed for a reproducible stratified split.",
    )
    return parser


def main() -> int:
    """Download, validate, split, and save the dataset."""
    args = create_parser().parse_args()
    if not 0 < args.test_size < 1:
        print("Error: --test-size must be between 0 and 1.", file=sys.stderr)
        return EXIT_FAILURE

    try:
        dataset = load_dataset()
        feature_columns = validate_dataset(dataset)
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

        print(f"Dataset: {len(dataset):,} rows, {len(feature_columns)} features")
        print(f"Target: {TARGET_COLUMN} ({dataset[TARGET_COLUMN].nunique()} classes)")
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