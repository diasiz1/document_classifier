"""GPU model 5: TF-IDF + feed-forward neural network in PyTorch (GPU version of models/mlp).

Training mirrors sklearn's MLPClassifier (Adam, L2 penalty `alpha`, stop after 10 epochs without the training
loss improving by 1e-4) and adds dropout. The trained weights are copied into an sklearn MLPClassifier for the
ONNX export, so the exported model is exactly the network trained on the GPU.
The TF-IDF runs on the CPU with sklearn (a second at this data size) so this package only needs PyTorch.
"""
from functools import cache
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

from models import common as cpu
from models_gpu import common

ARTIFACTS = Path(__file__).parent / "artifacts"
ONNX_PATH = ARTIFACTS / "model.onnx"
META_PATH = ARTIFACTS / "meta.json"

TFIDF_GRID = {
    **common.TFIDF_GRID,
    # the first layer has max_features x hidden weights, so the vocabulary must be capped
    "ngram_range": [(1, 1), (1, 2)],
    "max_features": [2_000, 5_000, 10_000, 20_000],
}
N_TFIDF = 10
CLF_GRID = {
    "hidden_layer_sizes": [(64,), (128,), (256,), (512,), (256, 128), (512, 256)],
    "activation": ["relu", "tanh"],
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],  # L2 regularization
    "dropout": [0.0, 0.2, 0.5],  # not in sklearn; a strong regularizer for wide first layers
    "learning_rate_init": [3e-4, 1e-3, 3e-3],
    "batch_size": [32, 64, 128, 256],
    "max_iter": [50, 100, 200],  # epochs
}
N_CLF = 20  # 200 candidates x 5 folds; a network trains in a few seconds on a GPU
SAMPLE_WEIGHT = True  # weighted cross-entropy, same as MLPClassifier.fit(..., sample_weight)
TOL = 1e-4  # sklearn MLPClassifier defaults
N_ITER_NO_CHANGE = 10


def _torch():
    try:
        import torch
    except ImportError:
        raise SystemExit("PyTorch is not installed: https://pytorch.org/get-started/locally/")
    return torch


@cache
def device():
    torch = _torch()
    if not torch.cuda.is_available():
        print("warning: no CUDA GPU found, training the network on the CPU")
        return torch.device("cpu")
    return torch.device("cuda")


def to_tensor(X):
    torch = _torch()
    return torch.as_tensor(X.toarray(), dtype=torch.float32, device=device())


def vectorize(params, train_chunks, other_chunks=None):
    tfidf = cpu.build_tfidf().set_params(**params)
    X = to_tensor(tfidf.fit_transform(train_chunks))
    X_other = None if other_chunks is None else to_tensor(tfidf.transform(other_chunks))
    return tfidf, X, X_other


def build_net(n_features, n_classes, params):
    nn = _torch().nn
    activation = {"relu": nn.ReLU, "tanh": nn.Tanh}[params["activation"]]
    layers, n_in = [], n_features
    for n_out in params["hidden_layer_sizes"]:
        layers += [nn.Linear(n_in, n_out), activation(), nn.Dropout(params["dropout"])]
        n_in = n_out
    layers.append(nn.Linear(n_in, n_classes))  # softmax is applied in the loss / at prediction
    return nn.Sequential(*layers)


def fit_net(X, y, params, n_classes, sample_weight=None):
    torch = _torch()
    F = torch.nn.functional
    torch.manual_seed(common.SEED)
    net = build_net(X.shape[1], n_classes, params).to(X.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=params["learning_rate_init"])
    weight_matrices = [m.weight for m in net if isinstance(m, torch.nn.Linear)]
    y = torch.as_tensor(y, dtype=torch.long, device=X.device)
    w = torch.ones(len(y), device=X.device) if sample_weight is None else \
        torch.as_tensor(sample_weight, dtype=torch.float32, device=X.device)

    n, batch_size = len(y), min(params["batch_size"], len(y))
    best_loss, no_improvement = np.inf, 0
    net.train()
    for _ in range(params["max_iter"]):
        epoch_loss = torch.zeros((), device=X.device)
        for batch in torch.randperm(n, device=X.device).split(batch_size):
            ce = F.cross_entropy(net(X[batch]), y[batch], reduction="none")
            loss = (ce * w[batch]).sum() / w[batch].sum()
            # sklearn's L2 term: alpha / 2 * sum of squared weights (not biases) / batch size
            loss = loss + 0.5 * params["alpha"] * sum((W ** 2).sum() for W in weight_matrices) / len(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach() * len(batch)
        epoch_loss = epoch_loss.item() / n
        no_improvement = no_improvement + 1 if epoch_loss > best_loss - TOL else 0
        best_loss = min(best_loss, epoch_loss)
        if no_improvement >= N_ITER_NO_CHANGE:
            break
    net.eval()
    return net


def predict_proba(net, X):
    torch = _torch()
    with torch.no_grad():
        return torch.softmax(net(X), dim=1).double().cpu().numpy()


def fit_proba(X_train, y_train, X_val, params, n_classes, sample_weight=None):
    return predict_proba(fit_net(X_train, y_train, params, n_classes, sample_weight), X_val)


def to_sklearn(net, params, classes):
    """An sklearn MLPClassifier holding the network's weights (dropout is a no-op at prediction time)."""
    linears = [m for m in net if isinstance(m, _torch().nn.Linear)]
    mlp = MLPClassifier(hidden_layer_sizes=params["hidden_layer_sizes"], activation=params["activation"])
    mlp.coefs_ = [m.weight.detach().cpu().double().numpy().T for m in linears]  # torch stores (out, in)
    mlp.intercepts_ = [m.bias.detach().cpu().double().numpy() for m in linears]
    mlp.n_layers_ = len(linears) + 1
    mlp.n_outputs_ = len(classes)
    mlp.out_activation_ = "softmax"
    mlp.classes_ = classes
    mlp.n_features_in_ = mlp.coefs_[0].shape[0]
    mlp._label_binarizer = LabelBinarizer().fit(classes)
    return mlp


def final_pipeline(tfidf_params, clf_params, chunks, y, classes, sample_weight=None):
    if len(classes) < 3:
        raise SystemExit("the MLP export assumes 3+ classes (sklearn uses a single logistic output for 2)")
    tfidf, X, _ = vectorize(tfidf_params, chunks)
    net = fit_net(X, y, clf_params, len(classes), sample_weight)
    pipeline = Pipeline([("tfidf", tfidf), ("clf", to_sklearn(net, clf_params, classes))])
    common.check_same_proba(predict_proba(net, X[:200]), pipeline.predict_proba(chunks[:200]),
                            "PyTorch vs exported sklearn")
    return pipeline
