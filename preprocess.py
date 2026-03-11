"""Turn coffee metadata and reviews into features for classification.

Learned statistics and the text vocabulary are fitted only on the training
split. Inputs use 100g_USD; USD_100g is also accepted as a price column alias.
"""

from __future__ import annotations

import argparse
import json
import string
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROAST_LEVELS = {
    "Light": 1,
    "Medium-Light": 2,
    "Medium": 3,
    "Medium-Dark": 4,
    "Dark": 5,
}
REQUIRED_COLUMNS = {"roaster", "roast", "origin", "100g_USD", "review"}
STEMMER = PorterStemmer()
PUNCTUATION = str.maketrans("", "", string.punctuation)


@dataclass
class PreprocessingResult:
    """Aligned sparse matrices and their fitted transformation."""

    train: sparse.csr_matrix
    test: sparse.csr_matrix
    feature_names: list[str]
    preprocessor: ColumnTransformer


def clean_review(text: str) -> str:
    """Lowercase, remove punctuation/stop words, and apply Porter stemming.

    Use scikit-learn's bundled English stop words so no NLTK corpus downloads
    are needed.
    """
    tokens = text.lower().translate(PUNCTUATION).split()
    return " ".join(
        STEMMER.stem(token) for token in tokens if token not in ENGLISH_STOP_WORDS
    )


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate inputs and clean features while retaining the original row order.

    Missing roast values default to Medium-Dark. Existing roast values are
    preserved. IDs, saved CSV indices, and labels are omitted.
    """
    frame = frame.copy().reset_index(drop=True)
    if "USD_100g" in frame.columns:
        if "100g_USD" in frame.columns:
            raise ValueError("Provide only one price column: USD_100g or 100g_USD.")
        frame = frame.rename(columns={"USD_100g": "100g_USD"})

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Input data must contain at least one row.")

    price = pd.to_numeric(frame["100g_USD"], errors="raise")
    if not np.isfinite(price.to_numpy(dtype=float)).all() or (price < 0).any():
        raise ValueError("Coffee prices must be finite, nonnegative numbers.")

    roast = frame["roast"].astype("string").str.strip().replace("", pd.NA)
    roast = roast.fillna("Medium-Dark")
    unknown = set(roast.unique()).difference(ROAST_LEVELS)
    if unknown:
        raise ValueError(f"Unknown roast values: {', '.join(sorted(unknown))}")

    features = pd.DataFrame(index=frame.index)
    for column in ("roaster", "origin"):
        features[column] = (
            frame[column].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        )
    features["100g_USD"] = price
    features["roast_num"] = roast.map(ROAST_LEVELS).astype(int)
    features["review"] = frame["review"].fillna("").astype(str).map(clean_review)
    return features


def preprocess(train: pd.DataFrame, test: pd.DataFrame) -> PreprocessingResult:
    """Fit on training rows and apply the same transformation to test rows.

    For validation or cross-validation, call this separately within each split
    so its validation rows cannot influence feature fitting.
    """
    train_features = prepare_features(train)
    test_features = prepare_features(test)
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), ["100g_USD", "roast_num"]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["roaster", "origin"],
            ),
            ("text", TfidfVectorizer(), "review"),
        ],
        sparse_threshold=1.0,
    )
    train_matrix = sparse.csr_matrix(preprocessor.fit_transform(train_features))
    test_matrix = sparse.csr_matrix(preprocessor.transform(test_features))
    return PreprocessingResult(
        train=train_matrix,
        test=test_matrix,
        feature_names=preprocessor.get_feature_names_out().tolist(),
        preprocessor=preprocessor,
    )


def preprocess_files(
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    labels_path: Path | None = None,
) -> PreprocessingResult:
    """Read CSVs and export sparse features, names, and a fitted transformer."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    labels = None
    if labels_path is not None:
        labels = pd.read_csv(labels_path)
        if list(labels.columns) != ["class"]:
            raise ValueError("Labels CSV must contain exactly one column named class.")
        if len(labels) != len(train) or labels["class"].isna().any():
            raise ValueError("Labels must have one nonmissing value per training row.")

    result = preprocess(train, test)
    output_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output_dir / "X_train.npz", result.train)
    sparse.save_npz(output_dir / "X_test.npz", result.test)
    (output_dir / "feature_names.json").write_text(
        json.dumps(result.feature_names, indent=2) + "\n", encoding="utf-8"
    )
    # This transformer contains only scikit-learn objects. For new CSV rows,
    # call prepare_features before passing them to the loaded transformer.
    joblib.dump(result.preprocessor, output_dir / "preprocessor.joblib")
    if labels is not None:
        labels.to_csv(output_dir / "y_train.csv", index=False)
    return result


def main() -> None:
    """Run preprocessing from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path, help="Training features CSV")
    parser.add_argument("--test", required=True, type=Path, help="Test features CSV")
    parser.add_argument("--labels", type=Path, help="Optional training labels CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    try:
        result = preprocess_files(args.train, args.test, args.output_dir, args.labels)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Training features: {result.train.shape}; test features: {result.test.shape}")
    print(f"Saved preprocessing artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
