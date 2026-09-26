"""Compare every trained model (models/*/artifacts and models_gpu/*/artifacts) on data/test.json.

Run from the project root:  python -m models.compare
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from models.common import TEST_PATH, OnnxModel, load

ROOT = Path(__file__).parent.parent
ARTIFACT_DIRS = sorted(ROOT.glob("models/*/artifacts")) + sorted(ROOT.glob("models_gpu/*/artifacts"))


def main():
    texts, labels = load(TEST_PATH)
    print(f"{'model':<20}{'cv_f1':>8}{'cv_logloss':>12}{'test_acc':>10}{'test_f1':>9}{'mean_conf':>11}{'sec':>7}")
    for artifacts in ARTIFACT_DIRS:
        name = f"{artifacts.parent.parent.name.removeprefix('models').strip('_') or 'cpu'}/{artifacts.parent.name}"
        onnx_path, meta_path = artifacts / "model.onnx", artifacts / "meta.json"
        if not onnx_path.exists() or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        onnx_model = OnnxModel(onnx_path, meta_path)
        start = perf_counter()
        proba = np.nan_to_num(onnx_model.predict_proba(texts), nan=-1)
        seconds = perf_counter() - start
        pred = onnx_model.classes[proba.argmax(axis=1)]
        print(f"{name:<20}{meta['cv_f1_macro']:>8.3f}{meta['cv_log_loss']:>12.4f}"
              f"{accuracy_score(labels, pred):>10.3f}{f1_score(labels, pred, average='macro', zero_division=0):>9.3f}"
              f"{proba.max(axis=1).mean():>11.3f}{seconds:>7.2f}")


if __name__ == "__main__":
    main()
