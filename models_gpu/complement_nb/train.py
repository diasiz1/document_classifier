"""Tune on the GPU, train and export GPU model 4 (cuML ComplementNB) to ONNX.

Run from the project root:  python -m models_gpu.complement_nb.train
"""
from models_gpu import common
from models_gpu.complement_nb import model

if __name__ == "__main__":
    common.train(model)
