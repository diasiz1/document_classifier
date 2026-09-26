"""GPU model 2: cuML TF-IDF + cuML LinearSVC with sigmoid calibration (GPU version of models/linear_svc).

cuML's LinearSVC has no predict_proba, so fit_proba reproduces
CalibratedClassifierCV(LinearSVC, method="sigmoid", cv=3): 3 inner fits on the GPU, one Platt sigmoid per class
on the held-out decision scores, normalized and averaged. That calibration ensemble can't be copied into sklearn,
so the winning setting is refit once with the CPU pipeline of models/linear_svc (seconds), on the GPU vocabulary.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from models.linear_svc import model as cpu_model
from models_gpu import common

ARTIFACTS = Path(__file__).parent / "artifacts"
ONNX_PATH = ARTIFACTS / "model.onnx"
META_PATH = ARTIFACTS / "meta.json"

# cuML's LinearSVC needs dense input: 50k features x a few thousand chunks is ~0.6 GB of GPU memory,
# an uncapped trigram vocabulary would be 10x that
TFIDF_GRID = {**common.TFIDF_GRID, "max_features": [5_000, 10_000, 20_000, 50_000]}
N_TFIDF = 40
CLF_GRID = [
    # C stops at 30: hinge loss with a larger C converges very slowly and hits max_iter
    {"penalty": ["l2"], "loss": ["squared_hinge", "hinge"], "C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]},
    # L1: sparse weights; only supported with squared_hinge
    {"penalty": ["l1"], "loss": ["squared_hinge"], "C": [0.1, 0.3, 1, 3, 10]},
]
N_CLF = None  # all 21 with every TF-IDF setting
SAMPLE_WEIGHT = False  # class_weight="balanced" instead
CALIBRATION_FOLDS = 3  # same as CalibratedClassifierCV(cv=3) in models/linear_svc


def vectorize(params, train_chunks, other_chunks=None):
    vec, X, X_other = common.cuml_vectorize(params, train_chunks, other_chunks)
    return vec, X.toarray(), None if X_other is None else X_other.toarray()


def build_clf(params):
    common.require_cuml()
    from cuml.svm import LinearSVC

    return LinearSVC(class_weight="balanced", max_iter=10_000, **params)


def platt(scores, target):
    """Platt scaling: fit P(target | score) = 1 / (1 + exp(a * score + b)), with Platt's smoothed targets."""
    n_pos = target.sum()
    n_neg = len(target) - n_pos
    t = np.where(target, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

    def loss(ab):
        z = ab[0] * scores + ab[1]
        p = expit(-z)
        # log loss and its gradient; logaddexp keeps it stable for large |z|
        value = np.sum(t * np.logaddexp(0, z) + (1 - t) * np.logaddexp(0, -z))
        d = t - p
        return value, np.array([np.dot(d, scores), d.sum()])

    a, b = minimize(loss, [0.0, np.log((n_neg + 1) / (n_pos + 1))], jac=True, method="L-BFGS-B").x
    return a, b


def _decision(clf, X):
    d = np.asarray(clf.decision_function(X), dtype=np.float64)
    return d[:, None] if d.ndim == 1 else d  # binary: one column for the positive class


def fit_proba(X_train, y_train, X_val, params, n_classes, sample_weight=None):
    import cupy as cp

    proba = np.zeros((X_val.shape[0], n_classes))
    inner = StratifiedKFold(n_splits=CALIBRATION_FOLDS)
    for fit_idx, cal_idx in inner.split(np.zeros(len(y_train)), y_train):
        fit_rows, cal_rows = cp.asarray(fit_idx), cp.asarray(cal_idx)
        weights = None if sample_weight is None else sample_weight[fit_idx]
        clf = build_clf(params).fit(X_train[fit_rows], y_train[fit_idx].astype(np.float32), sample_weight=weights)
        clf_classes = np.asarray(clf.classes_).astype(int)
        d_cal, d_val = _decision(clf, X_train[cal_rows]), _decision(clf, X_val)
        if d_cal.shape[1] == 1:  # binary: the score column belongs to the second class
            clf_classes = clf_classes[1:]
        p = np.zeros_like(proba)
        for col, k in enumerate(clf_classes):
            a, b = platt(d_cal[:, col], y_train[cal_idx] == k)
            p[:, k] = expit(-(a * d_val[:, col] + b))
        if len(clf_classes) == 1 and n_classes == 2:
            p[:, 1 - clf_classes[0]] = 1 - p[:, clf_classes[0]]
        total = p.sum(axis=1, keepdims=True)
        # like sklearn: normalize, and fall back to uniform when every sigmoid is 0
        p = np.divide(p, total, out=np.full_like(p, 1 / n_classes), where=total > 0)
        proba += p / CALIBRATION_FOLDS
    return proba


def final_pipeline(tfidf_params, clf_params, chunks, y, classes, sample_weight=None):
    vec, X, _ = common.cuml_vectorize(tfidf_params, chunks)
    tfidf = common.sklearn_tfidf_from_cuml(vec, tfidf_params, chunks, X)
    clf = cpu_model.build_model().named_steps["clf"]
    clf.set_params(**{f"estimator__{k}": v for k, v in clf_params.items()})
    clf.fit(tfidf.transform(chunks), classes[y])
    return Pipeline([("tfidf", tfidf), ("clf", clf)])
