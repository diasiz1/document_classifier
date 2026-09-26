"""Model 2: TF-IDF (word n-grams) + linear SVM (LinearSVC) with calibrated probabilities."""
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from models.common import SEED, TFIDF_GRID, build_tfidf

ARTIFACTS = Path(__file__).parent / "artifacts"
ONNX_PATH = ARTIFACTS / "model.onnx"
META_PATH = ARTIFACTS / "meta.json"

# searched in train.py
PARAM_GRID = [
    {
        **TFIDF_GRID,
        "clf__estimator__loss": ["squared_hinge", "hinge"],
        "clf__estimator__penalty": ["l2"],
        "clf__estimator__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100],
    },
    {
        # L1: sparse weights; LinearSVC supports it only with squared_hinge
        **TFIDF_GRID,
        "clf__estimator__loss": ["squared_hinge"],
        "clf__estimator__penalty": ["l1"],
        "clf__estimator__C": [0.1, 0.3, 1, 3, 10],
    },
]
N_ITER = 3000
SAMPLE_WEIGHT = False  # class_weight="balanced" instead


def build_model():
    # LinearSVC only gives decision scores; the calibration wrapper turns them into probabilities,
    # needed for log loss model selection and for averaging chunk probabilities per document.
    # Its internal 3-fold CV needs >= 3 chunks of every class in each training fold.
    # method is fixed to sigmoid (Platt scaling): isotonic exports to ONNX with up to 0.1 probability error
    # (float32 at its step boundaries) and "temperature" (sklearn 1.8+) is not supported by skl2onnx
    svc = LinearSVC(class_weight="balanced", max_iter=10_000, random_state=SEED)
    clf = CalibratedClassifierCV(svc, method="sigmoid", cv=3)
    return Pipeline([("tfidf", build_tfidf()), ("clf", clf)])
