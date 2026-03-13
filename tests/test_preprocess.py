"""Regression checks for leakage, alignment, validation, and saved artifacts."""

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from preprocess import prepare_features, preprocess, preprocess_files


def sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "coffee_id": [30, 10, 20],
            "roaster": ["Alpha", "Beta", "Alpha"],
            "roast": ["Light", None, "Dark"],
            "origin": ["Kenya", "Brazil", "Kenya"],
            "100g_USD": [5.0, 10.0, 15.0],
            "review": ["Sweet berries and chocolate", "Roasted cocoa", "Floral citrus"],
        },
        index=[5, 5, 9],
    )
    test = pd.DataFrame(
        {
            "coffee_id": [40, 50],
            "roaster": ["Unseen", "Alpha"],
            "roast": ["Medium", "Medium-Light"],
            "origin": ["Unseen", "Kenya"],
            "100g_USD": [1000.0, 20.0],
            "review": ["quasarword", None],
        }
    )
    return train, test


class PreprocessingTests(unittest.TestCase):
    def test_training_statistics_exclude_test_rows(self):
        train, test = sample_data()
        result = preprocess(train, test)
        fitted = result.preprocessor.named_transformers_
        np.testing.assert_allclose(fitted["numeric"].mean_, [10.0, 10 / 3])
        self.assertNotIn("quasarword", fitted["text"].vocabulary_)
        self.assertNotIn("Unseen", fitted["categorical"].categories_[0])
        self.assertEqual(result.train.shape[0], 3)
        self.assertEqual(result.test.shape, (2, len(result.feature_names)))
        self.assertTrue(sparse.isspmatrix_csr(result.train))
        self.assertTrue(np.isfinite(result.test.data).all())
        # Both categories in the first test row are unseen, so its one-hot
        # features should all be zero rather than creating new columns.
        categorical = [i for i, name in enumerate(result.feature_names)
                       if name.startswith("categorical__")]
        self.assertEqual(result.test[0, categorical].nnz, 0)

    def test_price_alias_indices_and_labels_cannot_change_features(self):
        train, test = sample_data()
        expected = preprocess(train, test)
        exported = train.rename(columns={"100g_USD": "USD_100g"}).copy()
        exported["Unnamed: 0"] = [100, 200, 300]
        exported["class"] = [0, 1, 0]
        actual = preprocess(exported, test)
        np.testing.assert_allclose(actual.train.toarray(), expected.train.toarray())
        self.assertEqual(actual.feature_names, expected.feature_names)
        prepared = prepare_features(exported)
        self.assertEqual(prepared["100g_USD"].tolist(), [5.0, 10.0, 15.0])
        self.assertEqual(prepared["roast_num"].tolist(), [1, 4, 5])
        self.assertIsNone(train.iloc[1]["roast"])

    def test_known_roast_is_preserved_for_expensive_coffee(self):
        train, _ = sample_data()
        train.loc[9, "roaster"] = "Difference Coffee"
        train.loc[9, "100g_USD"] = 111.11
        self.assertEqual(prepare_features(train).iloc[2]["roast_num"], 5)

    def test_invalid_schema_and_values_are_rejected(self):
        train, _ = sample_data()
        cases = [
            train.drop(columns="review"),
            train.iloc[:0],
            train.assign(roast="Unknown roast"),
            train.assign(**{"100g_USD": np.nan}),
            train.assign(**{"100g_USD": -1}),
            train.assign(**{"100g_USD": np.inf}),
            train.assign(USD_100g=1),
        ]
        for invalid in cases:
            with self.subTest(columns=list(invalid.columns)):
                with self.assertRaises(ValueError):
                    prepare_features(invalid)

    def test_export_round_trip_preserves_rows_and_transformer(self):
        train, test = sample_data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train.to_csv(root / "train.csv", index=False)
            test.to_csv(root / "test.csv", index=False)
            pd.DataFrame({"class": [1, 0, 1]}).to_csv(root / "labels.csv", index=False)
            output = root / "processed"
            result = preprocess_files(
                root / "train.csv", root / "test.csv", output, root / "labels.csv"
            )
            self.assertEqual(sparse.load_npz(output / "X_train.npz").shape, result.train.shape)
            self.assertEqual(
                json.loads((output / "feature_names.json").read_text()), result.feature_names
            )
            labels = pd.read_csv(output / "y_train.csv")
            self.assertEqual(labels["class"].tolist(), [1, 0, 1])
            restored = joblib.load(output / "preprocessor.joblib")
            actual = sparse.csr_matrix(restored.transform(prepare_features(test)))
            np.testing.assert_allclose(actual.toarray(), result.test.toarray())

    def test_label_mismatch_fails_before_export(self):
        train, test = sample_data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train.to_csv(root / "train.csv", index=False)
            test.to_csv(root / "test.csv", index=False)
            pd.DataFrame({"class": [1]}).to_csv(root / "labels.csv", index=False)
            output = root / "processed"
            with self.assertRaisesRegex(ValueError, "per training row"):
                preprocess_files(
                    root / "train.csv", root / "test.csv", output, root / "labels.csv"
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
