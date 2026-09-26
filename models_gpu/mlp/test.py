"""Evaluate the exported ONNX GPU model 5 (PyTorch MLP) on data/test.json.

Run from the project root:  python -m models_gpu.mlp.test
"""
from models import common
from models_gpu.mlp import model

if __name__ == "__main__":
    common.evaluate(model)
