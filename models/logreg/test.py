"""Evaluate the exported ONNX model 1 (LogisticRegression) on data/test.json.

Run from the project root:  python -m models.logreg.test
"""
from models import common
from models.logreg import model

if __name__ == "__main__":
    common.evaluate(model)
