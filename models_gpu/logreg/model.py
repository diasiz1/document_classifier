"""GPU model 1: cuML TF-IDF + cuML LogisticRegression (GPU version of models/logreg).

The GPU weights are copied into an sklearn LogisticRegression for the ONNX export, so the exported model
is exactly the one trained on the GPU.
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from models_gpu import common

ARTIFACTS = Path(__file__).parent / "artifacts"
ONNX_PATH = ARTIFACTS / "model.onnx"
META_PATH = ARTIFACTS / "meta.json"

TFIDF_GRID = common.TFIDF_GRID
N_TFIDF = 60
# cuML's quasi-Newton solver handles L2, L1 and elastic-net, so no solver choice is needed
CLF_GRID = [
    {"penalty": ["l2"], "C": [0.01, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]},
    # sparse weights: most stems get zero weight, only class-specific vocabulary is kept
    {"penalty": ["elasticnet"], "l1_ratio": [0.5], "C": [1, 3, 10, 100, 1000]},
    {"penalty": ["l1"], "C": [1, 3, 10, 100, 1000]},
]
N_CLF = None  # all 20 with every TF-IDF setting
SAMPLE_WEIGHT = False  # class_weight="balanced" instead

vectorize = common.cuml_vectorize


def build_clf(params):
    common.require_cuml()
    from cuml.linear_model import LogisticRegression as CumlLogisticRegression

    return CumlLogisticRegression(class_weight="balanced", max_iter=5000, **params)


def fit_proba(X_train, y_train, X_val, params, n_classes, sample_weight=None):
    clf = build_clf(params).fit(X_train, y_train.astype(np.float32), sample_weight=sample_weight)
    return common.expand_proba(clf.predict_proba(X_val), clf.classes_, n_classes)


def final_pipeline(tfidf_params, clf_params, chunks, y, classes, sample_weight=None):
    vec, X, _ = vectorize(tfidf_params, chunks)
    clf = build_clf(clf_params).fit(X, y.astype(np.float32), sample_weight=sample_weight)
    tfidf = common.sklearn_tfidf_from_cuml(vec, tfidf_params, chunks, X)

    coef = np.asarray(clf.coef_, dtype=np.float64)
    if coef.shape[0] == X.shape[1]:  # some cuML versions store (n_features, n_classes)
        coef = coef.T
    sk = LogisticRegression()
    sk.coef_ = coef
    sk.intercept_ = np.asarray(clf.intercept_, dtype=np.float64).ravel()
    sk.classes_ = classes[np.asarray(clf.classes_).astype(int)]
    sk.n_features_in_ = coef.shape[1]
    sk.n_iter_ = np.asarray([getattr(clf, "n_iter_", [0])]).ravel()[:1]
    pipeline = Pipeline([("tfidf", tfidf), ("clf", sk)])
    common.check_same_proba(clf.predict_proba(X[:200]), pipeline.predict_proba(chunks[:200]),
                            "cuML vs exported sklearn")
    return pipeline
