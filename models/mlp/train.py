"""Tune, train and export model 5 (MLPClassifier) to ONNX.

Run from the project root:  python -m models.mlp.train
"""
from models import common
from models.mlp import model

if __name__ == "__main__":
    common.train(model)
