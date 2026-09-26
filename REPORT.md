# Model 1 — Experiment Report

**Model:** preprocessing (tokenize → 6-char stemming → 200-token chunks, 50 overlap) → TF-IDF (word n-grams) → LogisticRegression, exported to ONNX.
**Goal:** check whether the model can learn the pattern of one document type (confidentiality agreement) before training on a diverse document set.

## Data

| Split | confidentiality_agreement | other | Chunks |
|---|---|---|---|
| train | 8 docs | 9 docs | 74 (62 / 12) |
| test | 2 docs | 3 docs | 19 (16 / 3) |

- **confidentiality_agreement** (`internal_docs/`): long (~1,400–1,700 words), templated agreements in Russian and Kazakh.
- **other** (`other_docs/`): short (50–420 words) government certificates (справки) from state bodies, mostly Kazakh/Russian, with noisy OCR.

## Results

Best hyperparameters (grid search of 324 settings, chosen by log loss):
`C=1000, ngram_range=(1,1), min_df=1, max_df=0.9, sublinear_tf=False`

| Metric | Value |
|---|---|
| CV f1_macro (train, grouped by document) | 1.00 |
| CV log loss | 0.012 |
| Test accuracy — chunk level (19) | 1.00 |
| Test accuracy — document level (5) | 1.00 |
| Test confidence per document | 0.985 – 0.999 |
| ONNX vs sklearn max difference | 8.7e-08 |

Choosing settings by F1 gave the same accuracy, but every probability was about 0.55–0.59. F1 can't tell those settings apart, so the search kept the most heavily regularized one it tried. Choosing by log loss fixed this.

## Interpretation

**What the experiment shows**
- The whole pipeline works: OCR → preprocessing → training → ONNX → inference. The ONNX model gives the same results as the scikit-learn one.
- Even with poor OCR quality, TF-IDF on shortened word stems cleanly separates the agreements from the certificates, in both Russian and Kazakh.
- Classifying chunks and then averaging per document works, and a single chunk is already enough to classify correctly.

**What it does *not* show**
- **The task is too easy to test pattern learning.** The two classes differ in length, vocabulary, layout and where they come from. Almost any text classifier would score 100% here, so the perfect score shows the classes are separable, not that the model generalizes.
- **The test set is tiny.** With 5 documents, one error changes accuracy by 20 points, so the results carry almost no statistical weight.
- **The confidence may be too high.** `C=1000` is the largest value in the search list. On perfectly separable data, log loss always favours the weakest regularization, so the ~0.99 confidence may be overstated for documents unlike these.
- **Chunk counts are unbalanced:** 62 agreement chunks vs 12 other chunks, because the certificates are short. `class_weight="balanced"` compensates for this during training.

## Issues found

- **Bug, fixed in `model1/train.py`:** the number of cross-validation folds was calculated from the wrong labels, so this run used only 2 folds instead of 5. Re-run `python -m model1.train` to get the correct CV scores.
- **OCR drops Kazakh-specific letters** (Қ→К, Ұ→У, Ғ→F, Ә→Э). The Cyrillic recognition model doesn't reliably output them, so Kazakh words don't match consistently between documents. This matters more once documents of different classes share vocabulary.
- **Duplicate document** in `other_docs/`: the certificate for КХ «Амир» appears twice in the training set.

## Recommendations for the diverse-set experiment

1. **Add confusable classes:** other agreements and contracts (employment contracts, NDAs with counterparties), which share legal vocabulary with the target class. This is the real test of pattern learning.
2. **Use at least 30–50 documents per class,** and repeat cross-validation or use several random splits to report a mean ± spread instead of a single split.
3. **Cap `C` (e.g. ≤ 100) or calibrate probabilities** (`CalibratedClassifierCV`) so confidence stays meaningful. Check the confidence of wrong predictions.
4. **Balance document lengths within each class,** or check that the model isn't just using length as a shortcut, e.g. by testing on short agreements and long "other" documents.
5. **Remove duplicates before splitting,** and keep near-identical documents on the same side of the split.
6. **Add an "unknown" confidence threshold** once calibrated, and measure how many out-of-distribution documents it rejects.
