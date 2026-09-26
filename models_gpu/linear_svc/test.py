"""Evaluate the exported ONNX GPU model 2 (cuML LinearSVC) on data/test.json.

Run from the project root:  python -m models_gpu.linear_svc.test
"""
from models import common
from models_gpu.linear_svc import model

if __name__ == "__main__":
    common.evaluate(model)
