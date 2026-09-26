"""Tune, train and export model 1 (LogisticRegression) to ONNX.

Run from the project root:  python -m models.logreg.train
"""
from models import common
from models.logreg import model

if __name__ == "__main__":
    common.train(model)
