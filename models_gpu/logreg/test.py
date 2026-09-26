"""Evaluate the exported ONNX GPU model 1 (cuML LogisticRegression) on data/test.json.

Run from the project root:  python -m models_gpu.logreg.test
"""
from models import common
from models_gpu.logreg import model

if __name__ == "__main__":
    common.evaluate(model)
