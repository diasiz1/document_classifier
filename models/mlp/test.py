"""Evaluate the exported ONNX model 5 (MLPClassifier) on data/test.json.

Run from the project root:  python -m models.mlp.test
"""
from models import common
from models.mlp import model

if __name__ == "__main__":
    common.evaluate(model)
