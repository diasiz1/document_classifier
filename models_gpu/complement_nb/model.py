"""GPU model 4: cuML TF-IDF + cuML Complement Naive Bayes (GPU version of models/complement_nb).

Naive Bayes is a closed-form count, so the winning setting is refit with sklearn on the GPU vocabulary for
the ONNX export; the result is the same model.
"""
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from models_gpu import common

ARTIFACTS = Path(__file__).parent / "artifacts"
ONNX_PATH = ARTIFACTS / "model.onnx"
META_PATH = ARTIFACTS / "meta.json"

# NB uses the term weights directly, so row normalization matters (None = raw tf-idf)
TFIDF_GRID = {**common.TFIDF_GRID, "norm": ["l2", "l1", None]}
N_TFIDF = 300  # NB fits take milliseconds, the TF-IDF is the whole cost
CLF_GRID = {
    "alpha": [1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0],  # additive smoothing
    "norm": [False, True],  # second normalization of the class weights
}
N_CLF = None
SAMPLE_WEIGHT = True  # ComplementNB has no class_weight

vectorize = common.cuml_vectorize


def fit_proba(X_train, y_train, X_val, params, n_classes, sample_weight=None):
    common.require_cuml()
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse
    from cuml.naive_bayes import ComplementNB as CumlComplementNB

    if sample_weight is not None:
        # cuML's fit has no sample_weight; NB only sums feature values per class, so scaling each row
        # by its weight gives exactly the counts of sklearn's fit(..., sample_weight)
        X_train = cpsparse.diags(cp.asarray(sample_weight, dtype=X_train.dtype)) @ X_train
    clf = CumlComplementNB(**params).fit(X_train, cp.asarray(y_train, dtype=cp.int32))
    return common.expand_proba(clf.predict_proba(X_val), clf.classes_, n_classes)


def final_pipeline(tfidf_params, clf_params, chunks, y, classes, sample_weight=None):
    vec, X, _ = common.cuml_vectorize(tfidf_params, chunks)
    tfidf = common.sklearn_tfidf_from_cuml(vec, tfidf_params, chunks, X)
    clf = ComplementNB(**clf_params).fit(tfidf.transform(chunks), classes[y], sample_weight=sample_weight)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])
