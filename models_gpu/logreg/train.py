"""Tune on the GPU, train and export GPU model 1 (cuML LogisticRegression) to ONNX.

Run from the project root:  python -m models_gpu.logreg.train
"""
from models_gpu import common
from models_gpu.logreg import model

if __name__ == "__main__":
    common.train(model)
