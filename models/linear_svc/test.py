"""Evaluate the exported ONNX model 2 (LinearSVC) on data/test.json.

Run from the project root:  python -m models.linear_svc.test
"""
from models import common
from models.linear_svc import model

if __name__ == "__main__":
    common.evaluate(model)
