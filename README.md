# Neural Steganalysis with Word-Level Attribution

A BERT-based detector for synonym-substitution linguistic steganography,
with three explainability methods (Integrated Gradients, LIME, SHAP)
**scored against ground truth** rather than inspected by eye.

## What this project does

1. **Embeds** hidden information in IMDB movie reviews by substituting
   words with context-appropriate synonyms, recording every substitution.
2. **Detects** whether a review has been modified, using a fine-tuned
   BERT classifier.
3. **Explains** each detection at word level, and — the part that
   distinguishes this work — **measures whether those explanations are
   correct**, by scoring the attributed words against the substitutions
   the encoder actually made.

Most explainability work stops at producing a heatmap. Because the
encoder records its own ground truth, the attributions here can be
scored with precision@k, recall@k, MAP and rank-AUC, each reported
against the random-ranking baseline for the same review.

## Why accuracy is reported with an embedding rate

A steganalysis accuracy figure is not interpretable on its own. Detecting
3 substituted words in a 230-word review is a fundamentally different
problem from detecting 30, and a single number hides which one was
measured.

`scripts/final_analysis.py` therefore reports detection **stratified by
the number of substitutions actually made**, producing a detection curve
rather than a point estimate. It also reports the accuracy ceiling implied
by ROC-AUC, which distinguishes two different situations: a detector
limited by its decision threshold, and one limited by how separable the
underlying signal is.

## Repository layout

```
src/
  config.py                     single source of truth for paths and
                                sequence length
  dataset.py                    builds balanced cover/stego splits
  text_utils.py                 tokenization preserving exact text
  steganography/
    synonym_stega.py            POS-aware synonym substitution with a
                                sentence-transformer similarity filter
  models/
    bert_detector.py            BERT sequence-classification wrapper
  explainability/
    integrated_gradients.py     IG with word-piece aggregation
  evaluation/
    metrics.py                  metrics, validation-based threshold
                                selection, bootstrap CIs, ceiling analysis
    localization.py             scores attributions against ground truth

scripts/
  prepare_dataset.py            deduplicate IMDB, stratified 70/15/15 split
  generate_synonym_stega.py     produce stego text plus substitution record
  build_detector_dataset.py     combine cover and stego into detector splits
  train_detector.py             fine-tune the detector
  score_split.py                save per-sample scores (the only GPU step
                                needed for analysis)
  final_analysis.py             all quantitative tables, CPU only
  explain_all.py                generate IG, LIME and SHAP explanations
  evaluate_localization.py      IG/LIME/SHAP comparison against ground truth
  compare_explainers.py         per-sample explainer comparison
  repair_metadata_ids.py        back-fill original_id into older metadata
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Place `IMDB Dataset.csv` in `data/raw/`.

## Reproducing the results

```bash
python -m scripts.prepare_dataset           # deduplicate and split
python -m scripts.generate_synonym_stega    # embed, recording ground truth
python -m scripts.build_detector_dataset    # assemble detector splits
python -m scripts.train_detector            # fine-tune (GPU)

python -m scripts.score_split --split validation
python -m scripts.score_split --split test
python -m scripts.final_analysis            # Tables 1-5

python -m scripts.explain_all --index N     # per-sample explanations
python -m scripts.evaluate_localization     # attribution vs ground truth
```

`--smoke-test` on `train_detector.py` trains on a small subset to verify
the pipeline. It writes to separate `_smoke` paths and never overwrites a
real checkpoint or report.

## Methodological notes

- **Sequence length** is defined once, in `src/config.py`. Training and
  evaluation tokenizing at different lengths silently degrades reported
  performance: measured on this corpus, the same checkpoint scored
  80.32% accuracy / 0.879 AUC at 256 tokens and 72.81% / 0.798 at 128.
- **Truncation is label noise.** At 256 tokens, 3.88% of stego reviews
  were byte-identical to their cover within the visible window while
  still carrying a positive label. At 512 that falls to 0.58%.
- **Decision thresholds** are selected on validation and applied unchanged
  to test. Selecting on test and reporting on test leaks the labels.
- **Splits are clean.** Reviews are deduplicated before splitting, splits
  are stratified by sentiment, and there is no `original_id` or exact-text
  overlap between train, validation and test.
- **Classes are balanced** (stego/cover ratio 0.988), so no class
  weighting is applied.
- **Attribution matching is by word identity, not token position.** A
  substituted word occurring elsewhere unmodified in the same review is
  counted wherever it ranks. `localization.py` reports
  `ambiguous_truth_words` per sample to bound this effect.

## Scope

This project covers detection of synonym-substitution steganography. It
does not identify which algorithm produced a given stego text; that
would require a second encoder, which is not implemented.
