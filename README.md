# Coffee Review Classification

A data mining project that predicts binary class labels from coffee reviews
using **Multinomial Naive Bayes**, with reusable preprocessing for numerical,
categorical, and text features.

The pipeline encodes roast intensity, scales price and roast, one-hot encodes
roaster and origin, and converts cleaned, Porter-stemmed reviews to TF-IDF.
It fits learned transformations on training rows only and handles unseen test
categories. Missing roast values default to **Medium-Dark**.

Requires **Python 3.10+**. The included CSVs contain **620 training records**,
their labels, and **267 test records**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python naive_bayes.py
```

The Naive Bayes experiment uses cleaned review word counts, an 80/20 stratified
validation split, and random seed **13**. It reports accuracy, precision, recall,
F1, and a confusion matrix, then refits on all labeled rows to predict the test
set. `artifacts/naive_bayes/` contains metrics, validation predictions, test
predictions in the input row order, and a saved model with its text vectorizer.

On the included data, this split gives **90.32% validation accuracy** and
**92.86% F1 for class 1** across 124 validation records.

To generate combined metadata and TF-IDF features separately:

```bash
python preprocess.py --train data/raw/X_train.csv \
  --test data/raw/X_test.csv --labels data/raw/y_train.csv
```

Outputs include sparse feature matrices, feature names, aligned labels, and the
fitted transformer in `data/processed/`. No NLTK corpus downloads are required.

Run the regression checks with:

```bash
python -m unittest discover -s tests -v
```
