"""Evaluate the exported ONNX model 4 (ComplementNB) on data/test.json.

Run from the project root:  python -m models.complement_nb.test
"""
from models import common
from models.complement_nb import model

if __name__ == "__main__":
    common.evaluate(model)
