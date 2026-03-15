"""Check holdout isolation, model refitting, prediction alignment, and reloads."""

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from naive_bayes import evaluate_model, load_labels, run_experiment
from preprocess import clean_review


def sample_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = [
        f"{'floral sweet' if i % 2 else 'bitter smoky'} unique{i}"
        for i in range(20)
    ]
    train = pd.DataFrame({"review": rows})
    labels = pd.DataFrame({"class": [i % 2 for i in range(20)]})
    test = pd.DataFrame({"review": ["floral sweet quasarword", "bitter smoky", None]})
    return train, labels, test


class NaiveBayesTests(unittest.TestCase):
    def test_validation_vocabulary_excludes_held_out_reviews(self):
        train, labels, _ = sample_data()
        result = evaluate_model(train["review"], labels["class"])
        vocabulary = result.model.named_steps["vectorizer"].vocabulary_
        for row in result.predictions["row_index"]:
            token = clean_review(f"unique{row}")
            self.assertNotIn(token, vocabulary)
        self.assertEqual(result.metrics["training_rows"], 16)
        self.assertEqual(result.metrics["validation_rows"], 4)
        expected = labels.iloc[result.predictions["row_index"]]["class"].tolist()
        self.assertEqual(result.predictions["actual_class"].tolist(), expected)
        actual = result.predictions["actual_class"]
        predicted = result.predictions["predicted_class"]
        self.assertEqual(result.metrics["f1"], f1_score(actual, predicted))
        self.assertEqual(
            result.metrics["confusion_matrix"],
            confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        )

    def test_validation_is_reproducible(self):
        train, labels, _ = sample_data()
        first = evaluate_model(train["review"], labels["class"])
        second = evaluate_model(train["review"], labels["class"])
        self.assertEqual(first.metrics, second.metrics)
        pd.testing.assert_frame_equal(first.predictions, second.predictions)

    def test_final_model_uses_all_labels_and_predicts_test_rows_in_order(self):
        train, labels, test = sample_data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, frame in [("train", train), ("labels", labels), ("test", test)]:
                frame.to_csv(root / f"{name}.csv", index=False)
            output = root / "artifacts"
            metrics = run_experiment(
                root / "train.csv", root / "labels.csv", root / "test.csv", output
            )
            model = joblib.load(output / "model.joblib")
            self.assertEqual(model.named_steps["classifier"].class_count_.sum(), len(train))
            vocabulary = model.named_steps["vectorizer"].vocabulary_
            self.assertNotIn("quasarword", vocabulary)
            for i in range(len(train)):
                self.assertIn(clean_review(f"unique{i}"), vocabulary)
            predictions = pd.read_csv(output / "predictions.csv")
            self.assertEqual(list(predictions.columns), ["class"])
            expected = model.predict(test["review"].fillna(""))
            np.testing.assert_array_equal(predictions["class"], expected)
            self.assertEqual(predictions["class"].iloc[:2].tolist(), [1, 0])
            self.assertEqual(metrics["test_rows"], len(test))
            self.assertEqual(json.loads((output / "metrics.json").read_text()), metrics)
            validation = pd.read_csv(output / "validation_predictions.csv")
            self.assertEqual(len(validation), metrics["validation_rows"])

    def test_invalid_labels_fail_before_artifacts_are_written(self):
        train, labels, test = sample_data()
        invalid_labels = [
            labels.iloc[:-1],
            labels.assign(**{"class": np.nan}),
            labels.assign(**{"class": 2}),
            labels.rename(columns={"class": "target"}),
        ]
        for frame in invalid_labels:
            with self.subTest(columns=list(frame.columns)):
                with self.assertRaises(ValueError):
                    load_labels(frame, len(train))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train.to_csv(root / "train.csv", index=False)
            labels.iloc[:-1].to_csv(root / "labels.csv", index=False)
            test.to_csv(root / "test.csv", index=False)
            output = root / "artifacts"
            with self.assertRaisesRegex(ValueError, "per training row"):
                run_experiment(
                    root / "train.csv", root / "labels.csv", root / "test.csv", output
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
