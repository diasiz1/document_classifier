"""Model 5: TF-IDF (word n-grams) + feed-forward neural network (MLPClassifier)."""
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from models.common import SEED, TFIDF_GRID, build_tfidf

ARTIFACTS = Path(__file__).parent / "artifacts"
ONNX_PATH = ARTIFACTS / "model.onnx"
META_PATH = ARTIFACTS / "meta.json"

# searched in train.py
PARAM_GRID = {
    **TFIDF_GRID,
    # the first layer has max_features x hidden weights, so the vocabulary must be capped:
    # 20k x 512 is already ~10M weights (~40 MB ONNX) learned from a few thousand chunks
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_features": [2_000, 5_000, 10_000, 20_000],
    "clf__hidden_layer_sizes": [(64,), (128,), (256,), (512,), (256, 128)],
    "clf__activation": ["relu", "tanh"],
    "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],  # L2 regularization, the main guard against overfitting
    "clf__learning_rate_init": [3e-4, 1e-3, 3e-3],
    "clf__batch_size": [32, 64, 128],
    "clf__max_iter": [50, 100, 200],  # epochs; fewer epochs also regularize
}
N_ITER = 60  # every candidate trains a network once per fold, so fewer candidates than the linear models
SAMPLE_WEIGHT = True  # MLPClassifier has no class_weight


def build_model():
    # early_stopping stays off: its internal stratified split fails for classes with one training chunk
    clf = MLPClassifier(solver="adam", n_iter_no_change=10, random_state=SEED)
    return Pipeline([("tfidf", build_tfidf()), ("clf", clf)])
