"""Code shared by all model packages: data loading, hyperparameter search, ONNX export, evaluation.

Every model package (models/<name>/) has a model.py that defines:
    ARTIFACTS, ONNX_PATH, META_PATH  where the exported model is saved
    PARAM_GRID                       hyperparameters to search (a dict, or a list of dicts)
    N_ITER                           None: try every setting in PARAM_GRID; int: N_ITER rounds of Bayesian
                                     optimization (Optuna TPE) over the ranges PARAM_GRID spans, see suggest()
    SAMPLE_WEIGHT                    True for classifiers without class_weight: pass balanced sample weights
    build_model()                    an untrained Pipeline([("tfidf", ...), ("clf", ...)])
and thin train.py / test.py that call train() / evaluate() below.
"""
import json
import os
import warnings
from collections import Counter
from pathlib import Path, PureWindowsPath
from time import perf_counter

import numpy as np
import onnxruntime as ort
from joblib import Parallel, delayed
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

import preprocess

# override with the DATA_DIR environment variable, e.g. DATA_DIR=/kaggle/input/docs-data on Kaggle
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
TRAIN_PATH = DATA_DIR / "train.json"
TEST_PATH = DATA_DIR / "test.json"
SEED = 67

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


def _suggest_value(trial, name, values):
    values = list(values)
    if len(values) == 1:
        return values[0]
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        # numbers become a range, so TPE can model them and try values between the listed ones;
        # log scale for ranges like C = 0.01..100
        lo, hi = min(values), max(values)
        log = lo > 0 and hi / lo >= 10
        if all(isinstance(v, int) for v in values):
            return trial.suggest_int(name, lo, hi, log=log)
        return trial.suggest_float(name, lo, hi, log=log)
    # everything else (tuples, strings, None, bools) is categorical; Optuna stores the index since it
    # only accepts primitive choices
    return values[trial.suggest_categorical(name, list(range(len(values))))]


def suggest(trial, grid):
    """Sample one setting from a PARAM_GRID-style grid (a dict, or a list of dicts) for an Optuna trial.

    Lists of numbers are searched as ranges between their smallest and largest value, other lists as categories.
    For a list of dicts the trial first picks a dict; a key whose values differ between dicts gets a separate
    Optuna parameter per dict (e.g. C of the lbfgs and saga branches), shared keys are learned across dicts.
    """
    grids = grid if isinstance(grid, list) else [grid]
    b = trial.suggest_categorical("grid", list(range(len(grids)))) if len(grids) > 1 else 0
    params = {}
    for key, values in grids[b].items():
        same = all(list(g[key]) == list(values) for g in grids if key in g)
        params[key] = _suggest_value(trial, key if same else f"{b}:{key}", values)
    return params


def tpe_study():
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    # multivariate: models interactions (e.g. best C depends on max_features); group: handles the
    # conditional spaces of list-of-dict grids; constant_liar: trials evaluated in parallel don't all
    # get suggested the same point
    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True, group=True, constant_liar=True)
    return optuna.create_study(direction="minimize", sampler=sampler)


def _fit_score(estimator, params, chunks, labels, classes, tr, va, sample_weight):
    """(log loss, f1_macro) of one setting on one fold; NaNs if it fails to fit."""
    chunks, labels = np.asarray(chunks, dtype=object), np.asarray(labels)
    fit_params = {} if sample_weight is None else {"clf__sample_weight": sample_weight[tr]}
    try:
        est = clone(estimator).set_params(**params).fit(chunks[tr], labels[tr], **fit_params)
        proba = np.zeros((len(va), len(classes)))
        proba[:, np.searchsorted(classes, est.classes_)] = est.predict_proba(chunks[va])
    except Exception as e:
        print(f"  failed {params}: {e}")
        return np.nan, np.nan
    y = labels[va]
    return log_loss(y, proba, labels=classes), f1_score(y, classes[proba.argmax(axis=1)], average="macro")


def bayes_search(model, chunks, labels, doc_ids, cv, sample_weight):
    """N_ITER settings chosen by TPE; each is scored like GridSearchCV would (mean over the CV folds)."""
    from optuna.trial import TrialState

    study = tpe_study()
    folds = list(cv.split(chunks, labels, doc_ids))
    classes = np.unique(labels)
    estimator = model.build_model()
    # BO is sequential, so settings are asked in batches that fill the CPU (batch x folds fits in parallel)
    batch = max(1, (os.cpu_count() or 1) // len(folds))
    seen = {}  # TPE can suggest a setting again (categorical/int params); reuse its score
    start = perf_counter()
    with Parallel(n_jobs=-1) as parallel:
        while len(study.trials) < model.N_ITER:
            trials = [study.ask() for _ in range(min(batch, model.N_ITER - len(study.trials)))]
            settings = [suggest(t, model.PARAM_GRID) for t in trials]
            new = list({repr(p): p for p in settings if repr(p) not in seen}.values())
            scores = parallel(delayed(_fit_score)(estimator, p, chunks, labels, classes, tr, va, sample_weight)
                              for p in new for tr, va in folds)
            for k, p in enumerate(new):
                seen[repr(p)] = np.mean(scores[k * len(folds):(k + 1) * len(folds)], axis=0)  # NaN if a fold failed
            for t, p in zip(trials, settings):
                loss, f1 = seen[repr(p)]
                t.set_user_attr("params", p)
                t.set_user_attr("f1_macro", float(f1))
                if np.isnan(loss):
                    study.tell(t, state=TrialState.FAIL)
                else:
                    study.tell(t, float(loss))
            done = [t for t in study.trials if t.value is not None]
            best = f"{study.best_value:.4f}" if done else "n/a"
            print(f"trial {len(study.trials)}/{model.N_ITER}, best CV log loss so far {best}, "
                  f"{perf_counter() - start:.0f}s")
    if not any(t.value is not None for t in study.trials):
        raise SystemExit("Every candidate failed.")
    best = study.best_trial
    fit_params = {} if sample_weight is None else {"clf__sample_weight": sample_weight}
    pipeline = clone(estimator).set_params(**best.user_attrs["params"]).fit(chunks, labels, **fit_params)
    return pipeline, best.user_attrs["params"], best.value, best.user_attrs["f1_macro"], len(seen)


def tune(model, chunks, labels, doc_ids):
    """Returns (pipeline refit on all chunks, best params, CV log loss, CV f1_macro, number of settings tried)."""
    # folds are split by document so chunks of one document never land in both train and validation
    doc_labels = dict(zip(doc_ids, labels))
    n_splits = min(5, min(Counter(doc_labels.values()).values()))
    if n_splits < 2:
        raise SystemExit("Need at least 2 training documents per class for cross-validation.")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    # same effect as class_weight="balanced" for classifiers that don't have that option
    sample_weight = compute_sample_weight("balanced", labels) if getattr(model, "SAMPLE_WEIGHT", False) else None
    # select by log loss: unlike F1 it rewards confident correct probabilities, so it can pick
    # among candidates that all classify perfectly (F1 ties would fall back to the first, weakest C)
    if model.N_ITER is None:
        search = GridSearchCV(model.build_model(), model.PARAM_GRID, cv=cv, n_jobs=-1, verbose=1,
                              scoring={"f1_macro": "f1_macro", "neg_log_loss": "neg_log_loss"},
                              refit="neg_log_loss")
        fit_params = {} if sample_weight is None else {"clf__sample_weight": sample_weight}
        search.fit(chunks, labels, groups=doc_ids, **fit_params)
        result = (search.best_estimator_, search.best_params_, -search.best_score_,
                  search.cv_results_["mean_test_f1_macro"][search.best_index_], len(search.cv_results_["params"]))
    else:
        # the full grids have thousands of settings; Bayesian optimization concentrates the budget on the
        # promising region instead of sampling it uniformly like RandomizedSearchCV
        result = bayes_search(model, chunks, labels, doc_ids, cv, sample_weight)
    print(f"best CV log loss: {result[2]:.4f}")
    print(f"best CV f1_macro: {result[3]:.4f}")
    print("best params:", result[1])
    return result


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
    pipeline, best_params, cv_log_loss, cv_f1_macro, n_candidates = tune(model, chunks, chunk_labels, doc_ids)

    model.ARTIFACTS.mkdir(exist_ok=True)
    onnx_diff = export_onnx(pipeline, chunks[:50], model.ONNX_PATH)
    meta = {
        "model": model.__name__.split(".")[-2],
        "classes": pipeline.classes_.tolist(),
        "best_params": {k: list(v) if isinstance(v, tuple) else v for k, v in best_params.items()},
        "cv_log_loss": float(cv_log_loss),
        "cv_f1_macro": float(cv_f1_macro),
        "search": "grid" if model.N_ITER is None else "bayesian (TPE)",
        "candidates_tried": n_candidates,
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
