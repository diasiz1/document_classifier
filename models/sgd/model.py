"""Model 3: TF-IDF (word n-grams) + linear model trained with stochastic gradient descent (SGDClassifier)."""
from pathlib import Path

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from models.common import SEED, TFIDF_GRID, build_tfidf

ARTIFACTS = Path(__file__).parent / "artifacts"
ONNX_PATH = ARTIFACTS / "model.onnx"
META_PATH = ARTIFACTS / "meta.json"

# searched in train.py
PARAM_GRID = {
    **TFIDF_GRID,
    # only these two losses have predict_proba: log_loss = logistic regression,
    # modified_huber = smoothed hinge (SVM-like, more tolerant of noisy chunks)
    "clf__loss": ["log_loss", "modified_huber"],
    "clf__penalty": ["l2", "l1", "elasticnet"],
    "clf__alpha": [1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2],
    "clf__l1_ratio": [0.15, 0.5, 0.85],  # only used with elasticnet
    "clf__learning_rate": ["optimal", "adaptive"],
    "clf__eta0": [0.01, 0.1],  # only used with adaptive
    "clf__average": [False, True],  # averaged SGD: smoother weights that often generalize better
}
N_ITER = 200
SAMPLE_WEIGHT = False  # class_weight="balanced" instead


def build_model():
    # the default loss ("hinge") has no predict_proba, so start from log_loss
    clf = SGDClassifier(loss="log_loss", max_iter=2000, tol=1e-4, class_weight="balanced", random_state=SEED)
    return Pipeline([("tfidf", build_tfidf()), ("clf", clf)])
