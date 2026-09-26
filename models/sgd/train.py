"""Tune, train and export model 3 (SGDClassifier) to ONNX.

Run from the project root:  python -m models.sgd.train
"""
from models import common
from models.sgd import model

if __name__ == "__main__":
    common.train(model)
