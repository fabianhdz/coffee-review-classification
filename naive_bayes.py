"""Train and evaluate Multinomial Naive Bayes on coffee review text.

The classifier uses word counts from cleaned reviews. Text cleaning is shared
with preprocess.py. Evaluation fits vocabulary on its training split only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocess import clean_review


@dataclass
class EvaluationResult:
    """Holdout model, metrics, and predictions with input row positions."""

    model: Pipeline
    metrics: dict
    predictions: pd.DataFrame


def build_model() -> Pipeline:
    """Bundle review cleaning, word counts, and Multinomial Naive Bayes."""
    return Pipeline(
        [
            ("vectorizer", CountVectorizer(preprocessor=clean_review)),
            ("classifier", MultinomialNB()),
        ]
    )


def load_reviews(frame: pd.DataFrame) -> pd.Series:
    """Read review text while preserving input row order."""
    if "review" not in frame.columns or frame.empty:
        raise ValueError("Features CSV must contain a review column and at least one row.")
    return frame["review"].fillna("").astype(str).reset_index(drop=True)


def load_labels(frame: pd.DataFrame, expected_rows: int) -> pd.Series:
    """Validate positional labels without joining on the target values."""
    if list(frame.columns) != ["class"]:
        raise ValueError("Labels CSV must contain exactly one column named class.")
    if len(frame) != expected_rows or frame["class"].isna().any():
        raise ValueError("Labels must have one nonmissing value per training row.")
    if set(frame["class"].unique()) != {0, 1}:
        raise ValueError("Training labels must contain both binary classes 0 and 1.")
    return frame["class"].astype(int).reset_index(drop=True)


def evaluate_model(
    reviews: pd.Series,
    labels: pd.Series,
    validation_size: float = 0.2,
    random_state: int = 13,
) -> EvaluationResult:
    """Evaluate on a stratified holdout; positive-class metrics refer to class 1."""
    reviews = load_reviews(pd.DataFrame({"review": reviews.reset_index(drop=True)}))
    labels = load_labels(pd.DataFrame({"class": labels.reset_index(drop=True)}), len(reviews))
    training_reviews, validation_reviews, training_labels, validation_labels = train_test_split(
        reviews, labels, test_size=validation_size, random_state=random_state, stratify=labels
    )
    model = build_model()
    model.fit(training_reviews, training_labels)
    predicted = model.predict(validation_reviews)
    metrics = {
        "model": "MultinomialNB",
        "features": "review word counts",
        "random_state": random_state,
        "validation_fraction": validation_size,
        "training_rows": len(training_reviews),
        "validation_rows": len(validation_reviews),
        "accuracy": float(accuracy_score(validation_labels, predicted)),
        "precision": float(precision_score(validation_labels, predicted, zero_division=0)),
        "recall": float(recall_score(validation_labels, predicted, zero_division=0)),
        "f1": float(f1_score(validation_labels, predicted, zero_division=0)),
        "positive_class": 1,
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": confusion_matrix(validation_labels, predicted, labels=[0, 1]).tolist(),
    }
    predictions = pd.DataFrame(
        {
            "row_index": validation_reviews.index,
            "actual_class": validation_labels.to_numpy(),
            "predicted_class": predicted,
        }
    )
    return EvaluationResult(model, metrics, predictions)


def run_experiment(
    train_path: Path,
    labels_path: Path,
    test_path: Path,
    output_dir: Path,
    validation_size: float = 0.2,
    random_state: int = 13,
) -> dict:
    """Evaluate, refit on all labeled rows, and export model and test predictions."""
    training_reviews = load_reviews(pd.read_csv(train_path))
    labels = load_labels(pd.read_csv(labels_path), len(training_reviews))
    test_reviews = load_reviews(pd.read_csv(test_path))
    evaluation = evaluate_model(training_reviews, labels, validation_size, random_state)

    # Evaluation stays separate from the final fit. Every labeled row is used
    # for the model that predicts the unlabeled test CSV.
    final_model = build_model()
    final_model.fit(training_reviews, labels)
    predictions = pd.DataFrame({"class": final_model.predict(test_reviews)})
    metrics = {
        **evaluation.metrics,
        "final_training_rows": len(training_reviews),
        "test_rows": len(test_reviews),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    evaluation.predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    # Preserve test input order and the dataset's single-column class format.
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    joblib.dump(final_model, output_dir / "model.joblib")
    return metrics


def main() -> None:
    """Run the Naive Bayes experiment from CSV inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/raw/X_train.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/raw/y_train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/raw/X_test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/naive_bayes"))
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=13)
    args = parser.parse_args()
    try:
        metrics = run_experiment(
            args.train, args.labels, args.test, args.output_dir,
            args.validation_size, args.random_state,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        f"Validation accuracy: {metrics['accuracy']:.4f}; "
        f"F1: {metrics['f1']:.4f}; "
        f"precision: {metrics['precision']:.4f}; recall: {metrics['recall']:.4f}"
    )
    print(f"Confusion matrix (rows=actual, columns=predicted; classes 0, 1): "
          f"{metrics['confusion_matrix']}")
    print(f"Saved model and {metrics['test_rows']} test predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
