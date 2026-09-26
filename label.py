"""OCR the source documents and write data/train.json and data/test.json.

Each JSON file is a list of {"file": ..., "text": ..., "label": ...}.
Folders and their labels are defined in SOURCES. PDFs are OCR'd (cached in data/ocr_cache/),
.txt files are read as-is.
"""
import json
from collections import Counter
from pathlib import Path
import os
from sklearn.model_selection import train_test_split

from ocr import ocr_pdf

SOURCES = {
    "docs": os.listdir("docs")
}
OUT_DIR = Path("data")
CACHE_DIR = OUT_DIR / "ocr_cache"
TEST_SIZE = 0.2
SEED = 42


def read_document(path):
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    cache = CACHE_DIR / path.parent.name / (path.stem + ".txt")
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    text = ocr_pdf(path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


def collect():
    records = []
    for folder, values in SOURCES.items():
        for label in values:
            files = sorted(p for p in Path(f"{folder}/{label}").glob("*") if p.suffix.lower() in {".pdf", ".txt"})
            if not files:
                print(f"warning: no documents in {folder}/")
            for path in files:
                print(f"reading {path}")
                records.append({"file": str(path), "text": read_document(path), "label": label})
    return records


def main():
    records = collect()
    counts = Counter(r["label"] for r in records)
    print("documents per class:", dict(counts))
    if len(counts) < 2:
        raise SystemExit("Need at least 2 classes; add unrelated documents to other_docs/.")

    train, test = train_test_split(
        records, test_size=TEST_SIZE, stratify=[r["label"] for r in records], random_state=SEED)

    OUT_DIR.mkdir(exist_ok=True)
    for name, split in (("train", train), ("test", test)):
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {path}: {len(split)} documents")


if __name__ == "__main__":
    main()
