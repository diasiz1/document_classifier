"""Tune, train and export model 2 (LinearSVC) to ONNX.

Run from the project root:  python -m models.linear_svc.train
"""
from models import common
from models.linear_svc import model

if __name__ == "__main__":
    common.train(model)
