# Business Entity Resolution — reproducible pipeline

Links Source-2 / Source-3 business records to the deduplicated Source-1 reference records.
The pipeline has four stages: blocking, a cheap re-ranker, a cross-fitted LightGBM pair model
plus a stacked context model, and an expected-F0.5 decision rule. See
`Documentation_template.md` at the root of the submission for the methodology.

## Environment

- Python 3.12, CPU only. Tested on 12 cores with 16 GB RAM (peak usage is about 11 GB).
- No network access is needed at any point, and no external data is used.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Data layout

`--data-dir` must point to the challenge `dataset/` folder:

```
dataset/
  train/train_source{1,2,3}.tsv, train/train_ground_truth.tsv
  test/test_source{1,2,3}.tsv
```

## Run end-to-end

From `code/business_entity_resolution/src`:

```bash
../.venv/bin/python -m ber.pipeline all \
    --data-dir /path/to/dataset \
    --work-dir /path/to/work \
    --output-dir /path/to/output
```

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The whole run takes
about 2.5 hours on a 12-core laptop. You can also run the steps one at a time, each with the
same flags:

| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `prepare` | Learns the Indic-script lexicon from training matches, then normalizes every record to parquet. | ~8 min |
| 2 | `rerank-train` | Trains the cheap candidate re-ranker on a sample of 150k training queries per country. | ~15 min |
| 3 | `candidates --split train` | Blocking (top-30 per query), re-ranking (keeps up to 6) and full pair features. | ~45 min |
| 4 | `candidates --split test` | The same for the test split. | ~40 min |
| 5 | `train` | 2-fold cross-fitted pair model, stacked context model, out-of-fold validation and decision tuning. Writes `work/models/decision.json`. | ~30 min |
| 6 | `predict` | Scores the test pairs, applies the tuned decision rule and writes both TSVs. | ~5 min |

Validate the output (run from the challenge `student_resource/` folder):

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Source layout

```
src/ber/
  config.py      paths, blocking and model hyper-parameters
  translit.py    rule-based transliteration for all 9 Brahmic scripts + learned lexicon
  normalize.py   name/address normalization, legal-form canonicalization, skeletons, phones
  prepare.py     TSV loading, lexicon learning, parallel normalization -> parquet
  blocking.py    13 hashed blocking-key families, IDF-weighted candidate scoring
  features.py    vectorized string similarities + IDF-weighted exact/fuzzy token overlap
  model.py       LightGBM training / prediction helpers
  decide.py      one-S1-per-record assignment + expected-F0.5 subset selection
  evaluate.py    the official macro F0.5 metric
  pipeline.py    CLI orchestration of all steps
```

## Determinism

Sampling and folds use fixed seeds with hash-based splits. LightGBM uses bagging with a fixed
seed, so reruns on the same hardware reproduce the outputs.
