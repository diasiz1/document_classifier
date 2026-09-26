"""GPU versions of the models in models/: hyperparameter search on the GPU, same ONNX artifacts as the CPU models.

Requirements (Linux or WSL2 with an NVIDIA GPU, e.g. a Kaggle notebook with a GPU accelerator):
    cuML models (logreg, linear_svc, complement_nb):  pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12
    MLP:                                              PyTorch with CUDA (falls back to CPU if there is no GPU)
    all:                                              pip install skl2onnx onnxruntime
Run from the project root:  python -m models_gpu.<name>.train   (DATA_DIR=... if data/ is elsewhere)

Search: sklearn's GridSearchCV would refit the TF-IDF for every candidate and fold. Here every TF-IDF setting is
vectorized once per fold and all classifier settings are scored on that matrix, so trying
N_TFIDF x N_CLF candidates costs only N_TFIDF x folds vectorizations. Folds, scoring (log loss, f1_macro)
and model selection are the same as in models/common.py.

Export: cuML models can't be converted to ONNX directly, so the final model is rebuilt as an ordinary sklearn
pipeline (TfidfVectorizer + classifier) with the GPU vocabulary and exported with skl2onnx. The ONNX file
therefore works with models/common.py's OnnxModel, the test scripts and models/compare.py unchanged.

Every package's model.py defines:
    ARTIFACTS, ONNX_PATH, META_PATH
    TFIDF_GRID, N_TFIDF   TfidfVectorizer settings (no "tfidf__" prefix); N_TFIDF random ones (None = all)
    CLF_GRID, N_CLF       classifier settings tried with every TF-IDF setting; N_CLF random ones (None = all)
    SAMPLE_WEIGHT         pass balanced sample weights to the classifier
    vectorize(params, train_chunks, other_chunks=None) -> (vectorizer, X_train, X_other) on the device
    fit_proba(X_train, y_train, X_val, params, n_classes, sample_weight) -> (n_val, n_classes) numpy probabilities
    final_pipeline(tfidf_params, clf_params, chunks, y, classes, sample_weight) -> fitted sklearn Pipeline
"""
import json
import warnings
from collections import Counter
from time import perf_counter

import numpy as np
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import ParameterGrid, ParameterSampler, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

import preprocess
from models import common as cpu

SEED = cpu.SEED
TRAIN_PATH = cpu.TRAIN_PATH
TEST_PATH = cpu.TEST_PATH
TFIDF_GRID = {k.removeprefix("tfidf__"): v for k, v in cpu.TFIDF_GRID.items()}
# TfidfVectorizer settings that still matter once the vocabulary is fixed (max_df, min_df, max_features only select it)
_TFIDF_WEIGHTING = ("ngram_range", "norm", "use_idf", "smooth_idf", "sublinear_tf", "binary")


def require_cuml():
    try:
        import cuml
    except ImportError:
        raise SystemExit("cuML is not installed. It needs Linux/WSL2 and an NVIDIA GPU:\n"
                         "  pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12")
    cuml.set_global_output_type("numpy")  # predictions and fitted attributes come back as numpy arrays
    return cuml


def cuml_vectorize(params, train_chunks, other_chunks=None):
    """TF-IDF on the GPU. Returns the cuML vectorizer and cupy sparse CSR matrices."""
    require_cuml()
    import cudf
    from cuml.feature_extraction.text import TfidfVectorizer

    # chunks are already lowercased, tokenized and space-separated (see preprocess.py)
    vec = TfidfVectorizer(lowercase=False, delimiter=" ", **params)
    X = vec.fit_transform(cudf.Series(list(train_chunks)))
    X_other = None if other_chunks is None else vec.transform(cudf.Series(list(other_chunks)))
    return vec, X, X_other


def _to_list(values):
    if hasattr(values, "to_pandas"):  # cudf
        values = values.to_pandas()
    return [str(v) for v in values]


def sklearn_tfidf_from_cuml(vec, params, chunks, X):
    """sklearn TfidfVectorizer with the vocabulary chosen by the cuML one (same column order), fitted on `chunks`."""
    names = vec.get_feature_names_out() if hasattr(vec, "get_feature_names_out") else vec.get_feature_names()
    # some cuML versions join n-grams with "_"; tokens from preprocess.py never contain "_", so this is safe
    vocabulary = [t.replace("_", " ") for t in _to_list(names)]
    weighting = {k: v for k, v in params.items() if k in _TFIDF_WEIGHTING}
    tfidf = cpu.build_tfidf().set_params(vocabulary=vocabulary, **weighting).fit(chunks)
    sample = X[:200]
    sample = sample.get() if hasattr(sample, "get") else sample  # cupy -> scipy
    diff = abs(sample - tfidf.transform(chunks[:200])).max()
    print(f"cuML vs sklearn TF-IDF max diff: {diff:.2e}")
    if diff > 1e-3:
        print("warning: the exported TF-IDF differs from the one used on the GPU")
    return tfidf


def expand_proba(proba, clf_classes, n_classes):
    """Probabilities for all classes (columns = class index), also when a class was missing from the training fold."""
    full = np.zeros((proba.shape[0], n_classes))
    full[:, np.asarray(clf_classes).astype(int)] = proba
    return full


def check_same_proba(expected, actual, what):
    diff = np.abs(np.asarray(expected) - np.asarray(actual)).max()
    print(f"{what} max probability diff: {diff:.2e}")
    if diff > 1e-3:
        print(f"warning: {what} differ")


def _settings(grid, n):
    all_settings = ParameterGrid(grid)
    if n is None or n >= len(all_settings):
        return list(all_settings)
    return list(ParameterSampler(grid, n_iter=n, random_state=SEED))


def search(model, chunks, y, doc_ids, n_classes):
    # folds are split by document so chunks of one document never land in both train and validation
    n_splits = min(5, min(Counter(dict(zip(doc_ids, y)).values()).values()))
    if n_splits < 2:
        raise SystemExit("Need at least 2 training documents per class for cross-validation.")
    folds = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(chunks, y, doc_ids))
    tfidf_settings = _settings(model.TFIDF_GRID, model.N_TFIDF)
    clf_settings = _settings(model.CLF_GRID, model.N_CLF)
    print(f"{len(tfidf_settings)} TF-IDF x {len(clf_settings)} classifier settings = "
          f"{len(tfidf_settings) * len(clf_settings)} candidates, {n_splits} folds")

    # scores[i, j, fold] = (log loss, f1_macro) of TF-IDF setting i + classifier setting j
    scores = np.full((len(tfidf_settings), len(clf_settings), n_splits, 2), np.nan)
    start = perf_counter()
    for i, tfidf_params in enumerate(tfidf_settings):
        for f, (tr, va) in enumerate(folds):
            try:
                _, X_tr, X_va = model.vectorize(tfidf_params, chunks[tr], chunks[va])
            except Exception as e:  # e.g. min_df/max_df leave no terms
                print(f"  skipped TF-IDF {tfidf_params}: {e}")
                break
            weights = compute_sample_weight("balanced", y[tr]) if model.SAMPLE_WEIGHT else None
            for j, clf_params in enumerate(clf_settings):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        proba = model.fit_proba(X_tr, y[tr], X_va, clf_params, n_classes, weights)
                except Exception as e:
                    print(f"  failed {clf_params}: {e}")
                    continue
                scores[i, j, f] = (log_loss(y[va], proba, labels=range(n_classes)),
                                   f1_score(y[va], proba.argmax(axis=1), average="macro"))
        done = scores[: i + 1, :, :, 0].mean(axis=2)
        best = np.nan if np.isnan(done).all() else np.nanmin(done)
        print(f"TF-IDF setting {i + 1}/{len(tfidf_settings)} done, best CV log loss so far {best:.4f}, "
              f"{perf_counter() - start:.0f}s")

    mean = scores.mean(axis=2)  # a candidate that failed on any fold stays NaN
    if np.isnan(mean[..., 0]).all():
        raise SystemExit("Every candidate failed.")
    order = np.argsort(np.where(np.isnan(mean[..., 0]), np.inf, mean[..., 0]), axis=None)
    ranked = [np.unravel_index(k, mean.shape[:2]) for k in order[:10]]
    top = [{"tfidf": tfidf_settings[i], "clf": clf_settings[j],
            "cv_log_loss": float(mean[i, j, 0]), "cv_f1_macro": float(mean[i, j, 1])} for i, j in ranked]
    best = top[0]
    print(f"best CV log loss: {best['cv_log_loss']:.4f}")
    print(f"best CV f1_macro: {best['cv_f1_macro']:.4f}")
    print("best params:", best["tfidf"], best["clf"])
    return best, top, int((~np.isnan(mean[..., 0])).sum())


def _jsonable(params, prefix):
    return {prefix + k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}


def train(model, train_path=TRAIN_PATH):
    """Search on the GPU, refit the best setting on all training chunks and export it to ONNX."""
    texts, labels = cpu.load(train_path)
    chunks, chunk_labels, doc_ids = preprocess.to_chunks(texts, labels)
    print(f"{len(texts)} documents -> {len(chunks)} chunks, per class: {dict(Counter(chunk_labels))}")
    classes, y = np.unique(chunk_labels, return_inverse=True)
    chunks = np.array(chunks, dtype=object)

    start = perf_counter()
    best, top, n_candidates = search(model, chunks, y, np.array(doc_ids), len(classes))
    weights = compute_sample_weight("balanced", y) if model.SAMPLE_WEIGHT else None
    pipeline = model.final_pipeline(best["tfidf"], best["clf"], chunks, y, classes, weights)

    model.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    onnx_diff = cpu.export_onnx(pipeline, list(chunks[:50]), model.ONNX_PATH)
    meta = {
        "model": "gpu/" + model.__name__.split(".")[-2],
        "classes": pipeline.classes_.tolist(),
        "best_params": {**_jsonable(best["tfidf"], "tfidf__"), **_jsonable(best["clf"], "clf__")},
        "cv_log_loss": best["cv_log_loss"],
        "cv_f1_macro": best["cv_f1_macro"],
        "candidates_tried": n_candidates,
        "train_seconds": round(perf_counter() - start, 1),
        "onnx_max_diff": float(onnx_diff),
        "top_candidates": [{"params": {**_jsonable(t["tfidf"], "tfidf__"), **_jsonable(t["clf"], "clf__")},
                            "cv_log_loss": t["cv_log_loss"], "cv_f1_macro": t["cv_f1_macro"]} for t in top],
        "preprocess": {"chunk_size": preprocess.CHUNK_SIZE, "chunk_overlap": preprocess.CHUNK_OVERLAP,
                       "stem_len": preprocess.STEM_LEN},
    }
    model.META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {model.ONNX_PATH} and {model.META_PATH}")
