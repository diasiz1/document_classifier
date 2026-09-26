"""Model 4: TF-IDF (word n-grams) + Complement Naive Bayes."""
from pathlib import Path

from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from models.common import TFIDF_GRID, build_tfidf

ARTIFACTS = Path(__file__).parent / "artifacts"
ONNX_PATH = ARTIFACTS / "model.onnx"
META_PATH = ARTIFACTS / "meta.json"

# searched in train.py. ComplementNB is the naive Bayes variant made for text with imbalanced classes:
# it estimates each class from all *other* classes' chunks, so small classes still get stable weights.
PARAM_GRID = {
    **TFIDF_GRID,
    # NB uses the term weights directly, so row normalization matters (None = raw tf-idf)
    "tfidf__norm": ["l2", "l1", None],
    "clf__alpha": [1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0],  # additive smoothing
    "clf__norm": [False, True],  # second normalization of the class weights
}
N_ITER = 3000
SAMPLE_WEIGHT = True  # ComplementNB has no class_weight


def build_model():
    return Pipeline([("tfidf", build_tfidf()), ("clf", ComplementNB())])
