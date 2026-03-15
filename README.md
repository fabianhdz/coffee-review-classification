# Coffee Review Classification

A data mining project that prepares coffee metadata and written reviews for
predicting binary class labels. The preprocessing combines numerical,
categorical, and text features into consistent training and test matrices.

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
python preprocess.py --train data/raw/X_train.csv \
  --test data/raw/X_test.csv --labels data/raw/y_train.csv
```

Outputs include sparse feature matrices, feature names, aligned labels, and the
fitted transformer in `data/processed/`. No NLTK corpus downloads are required.

Run the regression checks with:

```bash
python -m unittest discover -s tests -v
```
