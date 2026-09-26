"""Tune, train and export model 4 (ComplementNB) to ONNX.

Run from the project root:  python -m models.complement_nb.train
"""
from models import common
from models.complement_nb import model

if __name__ == "__main__":
    common.train(model)
