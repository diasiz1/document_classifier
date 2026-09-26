"""Evaluate the exported ONNX GPU model 4 (cuML ComplementNB) on data/test.json.

Run from the project root:  python -m models_gpu.complement_nb.test
"""
from models import common
from models_gpu.complement_nb import model

if __name__ == "__main__":
    common.evaluate(model)
