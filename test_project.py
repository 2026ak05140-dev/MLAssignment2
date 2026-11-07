"""Check Sensorless Drive parsing, validation, and saved model reproducibility.

Usage:
    python -m unittest test_project
"""

from __future__ import annotations

import io
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from prepare_data import CLASS_LABELS, FEATURE_COLUMNS, TARGET_COLUMN, load_dataset
from train_models import (
    build_models,
    evaluate_model_outputs,
    validate_data,
)

PROJECT_DIR = Path(__file__).resolve().parent


class SensorlessDriveTests(unittest.TestCase):
    """Verify the actual repository data and fitted pipelines."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.train = pd.read_csv(PROJECT_DIR / "train_data.csv")
        cls.test = pd.read_csv(PROJECT_DIR / "test_data.csv")

    def source_archive(self, invalid_target: bool = False) -> io.BytesIO:
        """Reconstruct the unnamed source columns to test parsing without a download."""
        source = pd.concat([self.train, self.test], ignore_index=True)
        if invalid_target:
            source[TARGET_COLUMN] = source[TARGET_COLUMN].astype(float)
            source.loc[0, TARGET_COLUMN] = 1.5
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "Sensorless_drive_diagnosis.txt",
                source.to_csv(sep=" ", header=False, index=False),
            )
        buffer.seek(0)
        return buffer

    def test_source_order_and_target_separation(self) -> None:
        with patch(
            "prepare_data.urllib.request.urlopen", return_value=self.source_archive()
        ):
            decoded = load_dataset()
        self.assertEqual(list(decoded.columns), [*FEATURE_COLUMNS, TARGET_COLUMN])
        self.assertEqual(set(decoded.Class), set(CLASS_LABELS))
        self.assertNotIn(TARGET_COLUMN, FEATURE_COLUMNS)
        expected = pd.concat([self.train, self.test], ignore_index=True)
        pd.testing.assert_frame_equal(decoded, expected)

    def test_noninteger_source_target_is_rejected(self) -> None:
        with (
            patch(
                "prepare_data.urllib.request.urlopen",
                return_value=self.source_archive(invalid_target=True),
            ),
            self.assertRaisesRegex(ValueError, "integer condition classes"),
        ):
            load_dataset()

    def test_split_and_schema(self) -> None:
        self.assertEqual(self.train.shape, (46807, 49))
        self.assertEqual(self.test.shape, (11702, 49))
        self.assertEqual(validate_data(self.train, self.test), FEATURE_COLUMNS)
        train_hashes = set(pd.util.hash_pandas_object(self.train, index=False))
        test_hashes = set(pd.util.hash_pandas_object(self.test, index=False))
        self.assertFalse(train_hashes & test_hashes)

    def test_invalid_inputs_are_rejected(self) -> None:
        cases = {
            "missing feature": self.test.drop(columns=[FEATURE_COLUMNS[0]]),
            "extra feature": self.test.assign(Extra=0),
            "unknown class": self.test.assign(Class="Unknown"),
            "noninteger class": self.test.assign(Class=1.5),
            "non-finite value": self.test.assign(Feature01=np.inf),
            "nonnumeric value": self.test.assign(Feature01="invalid"),
        }
        for name, invalid in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_data(self.train, invalid)

    def test_saved_models_and_training_statistics(self) -> None:
        for key, (name, _) in build_models().items():
            with self.subTest(model=name):
                model = joblib.load(PROJECT_DIR / "model" / f"{key}.joblib")
                self.assertEqual(list(model.feature_names_in_), FEATURE_COLUMNS)
                self.assertEqual(set(model.classes_), set(CLASS_LABELS))
                _, predictions = evaluate_model_outputs(
                    model,
                    self.test[FEATURE_COLUMNS].head(128),
                    self.test.Class.head(128),
                )
                np.testing.assert_array_equal(
                    predictions, model.predict(self.test[FEATURE_COLUMNS].head(128))
                )
                if "standardscaler" in model.named_steps:
                    scaler = model.named_steps["standardscaler"]
                    self.assertEqual(scaler.n_samples_seen_, len(self.train))
                    np.testing.assert_allclose(
                        scaler.mean_,
                        self.train[FEATURE_COLUMNS].mean().to_numpy(),
                        rtol=1e-12,
                    )

    def test_app_renders_every_focus_model(self) -> None:
        app = AppTest.from_file(str(PROJECT_DIR / "app.py")).run(timeout=600)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        saved_metrics = pd.read_csv(PROJECT_DIR / "model_metrics.csv").set_index(
            "ML Model Name"
        )
        pd.testing.assert_frame_equal(
            app.dataframe[2].value,
            saved_metrics,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        for name in app.selectbox[0].options:
            with self.subTest(model=name):
                app.selectbox[0].select(name).run(timeout=600)
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertEqual(len(app.metric), 6)
                self.assertEqual(len(app.dataframe[2].value), 5)
                self.assertEqual(len(app.dataframe[-1].value), 10)


if __name__ == "__main__":
    unittest.main()
