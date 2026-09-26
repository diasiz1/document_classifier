"""Tune on the GPU, train and export GPU model 5 (PyTorch MLP) to ONNX.

Run from the project root:  python -m models_gpu.mlp.train
"""
from models_gpu import common
from models_gpu.mlp import model

if __name__ == "__main__":
    common.train(model)
