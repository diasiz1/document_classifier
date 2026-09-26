"""Model 1: TF-IDF (word n-grams) + LogisticRegression, trained on document chunks."""
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from models.common import TFIDF_GRID, build_tfidf

ARTIFACTS = Path(__file__).parent / "artifacts"
ONNX_PATH = ARTIFACTS / "model.onnx"
META_PATH = ARTIFACTS / "meta.json"

# searched in train.py. Since sklearn 1.8 `penalty` is deprecated: l1_ratio=0 is L2, 1 is L1,
# in between is elastic-net (only the saga solver supports l1_ratio > 0)
PARAM_GRID = [
    {
        **TFIDF_GRID,
        "clf__solver": ["lbfgs"],
        "clf__l1_ratio": [0.0],
        "clf__C": [0.01, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000],
    },
    {
        # sparse weights: most stems get zero weight, only class-specific vocabulary is kept
        **TFIDF_GRID,
        "clf__solver": ["saga"],
        "clf__l1_ratio": [0.5, 1.0],
        "clf__C": [1, 10, 100, 1000],
    },
]
N_ITER = 1000
SAMPLE_WEIGHT = False  # class_weight="balanced" instead


def build_model():
    clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    return Pipeline([("tfidf", build_tfidf()), ("clf", clf)])
