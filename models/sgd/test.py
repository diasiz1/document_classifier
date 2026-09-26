"""Evaluate the exported ONNX model 3 (SGDClassifier) on data/test.json.

Run from the project root:  python -m models.sgd.test
"""
from models import common
from models.sgd import model

if __name__ == "__main__":
    common.evaluate(model)
