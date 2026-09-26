"""Tune on the GPU, train and export GPU model 2 (cuML LinearSVC) to ONNX.

Run from the project root:  python -m models_gpu.linear_svc.train
"""
from models_gpu import common
from models_gpu.linear_svc import model

if __name__ == "__main__":
    common.train(model)
