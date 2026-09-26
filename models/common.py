"""Code shared by all model packages: data loading, hyperparameter search, ONNX export, evaluation.

Every model package (models/<name>/) has a model.py that defines:
    ARTIFACTS, ONNX_PATH, META_PATH  where the exported model is saved
    PARAM_GRID                       hyperparameters to search (a dict, or a list of dicts)
    N_ITER                           None: try every setting in PARAM_GRID; int: try N_ITER random ones
    SAMPLE_WEIGHT                    True for classifiers without class_weight: pass balanced sample weights
    build_model()                    an untrained Pipeline([("tfidf", ...), ("clf", ...)])
and thin train.py / test.py that call train() / evaluate() below.
"""
import json
import os
from collections import Counter
from pathlib import Path, PureWindowsPath
from time import perf_counter

import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

import preprocess

# override with the DATA_DIR environment variable, e.g. DATA_DIR=/kaggle/input/docs-data on Kaggle
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
TRAIN_PATH = DATA_DIR / "train.json"
TEST_PATH = DATA_DIR / "test.json"
SEED = 42

# TF-IDF part of every PARAM_GRID. Not searched on purpose:
#   analyzer / token_pattern / lowercase / stop_words / strip_accents - text is already normalized,
#     tokenized, stemmed and stop-word filtered in preprocess.py; skl2onnx also only exports analyzer="word"
#   smooth_idf - only avoids division by zero, negligible effect next to min_df/max_df
#   norm - kept at "l2" for margin-based models (l1 shrinks every value and just shifts the best C);
#     naive Bayes overrides it
#   sublinear_tf - skl2onnx 1.20 exports it as log(1 + tf) instead of sklearn's 1 + log(tf), so the ONNX
#     model would disagree with the one tuned here (~1e-2 probability error). binary=True is a similar damping
TFIDF_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tfidf__min_df": [1, 2, 3, 5],  # absolute chunk counts
    "tfidf__max_df": [0.5, 0.7, 0.85, 1.0],  # with 26 classes, a term in >50% of chunks says little
    # 26 classes with bi/trigrams give a vocabulary of 10^5+ terms, most of them rare OCR noise.
    # max_features keeps only the most frequent ones: less overfitting, smaller and faster ONNX model
    "tfidf__max_features": [5_000, 10_000, 20_000, 50_000, None],
    "tfidf__use_idf": [True, False],
    "tfidf__binary": [False, True],
}


def build_tfidf():
    # Chunks arrive already lowercased and tokenized (see preprocess.py), so the vectorizer
    # only splits on spaces. This also keeps the pipeline convertible to ONNX.
    return TfidfVectorizer(token_pattern=r"\S+", lowercase=False)


def resolve(file):
    # records store Windows-style paths relative to the project root ("data\ocr_cache\...");
    # make them work on Linux too, relative to DATA_DIR
    parts = PureWindowsPath(file).parts
    if parts and parts[0] == "data":
        return DATA_DIR.joinpath(*parts[1:])
    return Path(*parts)


def load(path):
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    texts = []
    labels = []
    for r in records:
        with open(resolve(r['file']), encoding='utf-8') as f:
            texts.append(f.read())
        
        labels.append(r['label'])

    return texts, labels


def tune(model, chunks, labels, doc_ids):
    # folds are split by document so chunks of one document never land in both train and validation
    doc_labels = dict(zip(doc_ids, labels))
    n_splits = min(5, min(Counter(doc_labels.values()).values()))
    if n_splits < 2:
        raise SystemExit("Need at least 2 training documents per class for cross-validation.")
    # select by log loss: unlike F1 it rewards confident correct probabilities, so it can pick
    # among candidates that all classify perfectly (F1 ties would fall back to the first, weakest C)
    options = dict(
        scoring={"f1_macro": "f1_macro", "neg_log_loss": "neg_log_loss"}, refit="neg_log_loss",
        n_jobs=-1, verbose=1, cv=StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED))
    if model.N_ITER is None:
        search = GridSearchCV(model.build_model(), model.PARAM_GRID, **options)
    else:
        # the full grids have thousands of settings; a random sample of them finds a near-best one much faster
        search = RandomizedSearchCV(model.build_model(), model.PARAM_GRID, n_iter=model.N_ITER,
                                    random_state=SEED, **options)
    fit_params = {}
    if getattr(model, "SAMPLE_WEIGHT", False):
        # same effect as class_weight="balanced" for classifiers that don't have that option
        fit_params["clf__sample_weight"] = compute_sample_weight("balanced", labels)
    search.fit(chunks, labels, groups=doc_ids, **fit_params)
    print(f"best CV log loss: {-search.best_score_:.4f}")
    print(f"best CV f1_macro: {cv_f1(search):.4f}")
    print("best params:", search.best_params_)
    return search


def cv_f1(search):
    return search.cv_results_["mean_test_f1_macro"][search.best_index_]


def export_onnx(pipeline, sample, onnx_path):
    onx = to_onnx(
        pipeline,
        initial_types=[("input", StringTensorType([None, 1]))],
        options={TfidfVectorizer: {"separators": [" "]}, type(pipeline[-1]): {"zipmap": False}},
        target_opset=17,
    )
    onnx_path.write_bytes(onx.SerializeToString())

    # sanity check: ONNX output must match sklearn
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_proba = sess.run(None, {"input": np.array(sample, dtype=object).reshape(-1, 1)})[1]
    diff = np.abs(onnx_proba - pipeline.predict_proba(sample)).max()
    print(f"ONNX vs sklearn max probability diff: {diff:.2e}")
    if diff > 1e-2:  # ONNX runs in float32, small differences are expected
        print("warning: ONNX output differs from sklearn")
    return diff


def train(model, train_path=TRAIN_PATH):
    """Tune, refit on all training chunks and export `model` (a models/<name>/model.py module)."""
    texts, labels = load(train_path)
    chunks, chunk_labels, doc_ids = preprocess.to_chunks(texts, labels)
    print(f"{len(texts)} documents -> {len(chunks)} chunks, per class: {dict(Counter(chunk_labels))}")

    start = perf_counter()
    search = tune(model, chunks, chunk_labels, doc_ids)
    pipeline = search.best_estimator_  # refit on all training chunks

    model.ARTIFACTS.mkdir(exist_ok=True)
    onnx_diff = export_onnx(pipeline, chunks[:50], model.ONNX_PATH)
    meta = {
        "model": model.__name__.split(".")[-2],
        "classes": pipeline.classes_.tolist(),
        "best_params": {k: list(v) if isinstance(v, tuple) else v for k, v in search.best_params_.items()},
        "cv_log_loss": -search.best_score_,
        "cv_f1_macro": cv_f1(search),
        "candidates_tried": len(search.cv_results_["params"]),
        "train_seconds": round(perf_counter() - start, 1),
        "onnx_max_diff": float(onnx_diff),
        "preprocess": {"chunk_size": preprocess.CHUNK_SIZE, "chunk_overlap": preprocess.CHUNK_OVERLAP,
                       "stem_len": preprocess.STEM_LEN},
    }
    model.META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {model.ONNX_PATH} and {model.META_PATH}")


class OnnxModel:
    """Inference with an exported model: chunk-level and document-level probabilities."""

    def __init__(self, onnx_path, meta_path):
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.classes = np.array(json.loads(meta_path.read_text(encoding="utf-8"))["classes"])

    def chunk_proba(self, chunks):
        inputs = np.array(chunks, dtype=object).reshape(-1, 1)
        return self.session.run(None, {"input": inputs})[1]

    def predict_proba(self, texts):
        """Document-level probabilities: mean of chunk probabilities (NaN for empty documents)."""
        chunks, _, doc_ids = preprocess.to_chunks(texts)
        doc_ids = np.array(doc_ids)
        proba = np.full((len(texts), len(self.classes)), np.nan)
        if chunks:
            cp = self.chunk_proba(chunks)
            for i in np.unique(doc_ids):
                proba[i] = cp[doc_ids == i].mean(axis=0)
        return proba

    def predict(self, texts):
        return self.classes[np.nan_to_num(self.predict_proba(texts), nan=-1).argmax(axis=1)]


def evaluate(model, test_path=TEST_PATH):
    """Evaluate the exported ONNX model of `model` (a models/<name>/model.py module) on the test set."""
    texts, labels = load(test_path)
    onnx_model = OnnxModel(model.ONNX_PATH, model.META_PATH)

    chunks, chunk_labels, _ = preprocess.to_chunks(texts, labels)
    chunk_pred = onnx_model.classes[onnx_model.chunk_proba(chunks).argmax(axis=1)]
    print(f"=== chunk level ({len(chunks)} chunks) ===")
    print(classification_report(chunk_labels, chunk_pred, zero_division=0))

    start = perf_counter()
    proba = onnx_model.predict_proba(texts)
    print(f"------------------------------------- Performed in {perf_counter() - start}")
    pred = onnx_model.classes[np.nan_to_num(proba, nan=-1).argmax(axis=1)]
    print(f"=== document level ({len(texts)} documents) ===")
    print(classification_report(labels, pred, zero_division=0))
    print("confusion matrix (rows = true, cols = predicted):", onnx_model.classes.tolist())
    print(confusion_matrix(labels, pred, labels=onnx_model.classes))

    with open(test_path, encoding="utf-8") as f:
        files = [r["file"] for r in json.load(f)]
    print("\nper document:")
    for file, true, p, pr in zip(files, labels, pred, proba):
        mark = "ok " if true == p else "ERR"
        print(f"{mark} {np.nanmax(pr):.3f}  true={true:<40} pred={p:<40} {file}")
