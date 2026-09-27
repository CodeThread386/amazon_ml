# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We built a cascade: 13 families of hashed blocking keys, then a cheap LightGBM re-ranker, then a cross-fitted LightGBM pair model with ~70 features, then a structured decision rule. The data has a key property: every Source-2/3 record matches at most one Source-1 entity. We exploit it with an assignment step, then choose each entity's match set to maximise expected F0.5. Records in 8 Indic scripts are handled by a single transliterator and a lexicon learned from the training matches. No external data is used.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from exploring the training data:

**Structure of the matches**
- There are 7,638,365 true pairs, and every matched S2/S3 id appears exactly once. So every S2/S3 record belongs to **at most one** Source-1 entity.
- About 26% of S2/S3 records are distractors with no match.
- 5.6% of Source-1 entities are singletons with no match; 90% have between 2 and 6 matches.

**Name noise**
- Typos, character swaps and digit-for-letter substitutions ("Pr0vision").
- Added or removed legal forms (LLC, Pvt Ltd, SARL), shuffled word order, and extra words ("Center", "Trading", "Foundation").
- Honorific prefixes ("M/s", "Smt").
- Phone numbers embedded in the name.
- Domain-style names ("maurewilliamscolombier.com").
- DBA names unrelated to the legal name (these can only be matched through the address).

**Script noise**
- About 20% of Indian S2/S3 names are written in Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali, Malayalam, Gurmukhi or Odia.
- In addresses, native script appears only as the state name (16 distinct values).

**Address noise**
- Components are reordered, and street words are abbreviated or wrongly expanded ("ST" becomes "SAINT").
- Ordinals are corrupted ("45ND") and house numbers are zero-padded ("0684").
- Numbers are sometimes spelled out ("NINTH").
- Addresses can be missing entirely (about 3.4%), or contain `null` / `<NULL>` placeholders.
- States appear as full names or abbreviations; Indian landmarks and phone numbers appear inside addresses.

**Test-set shift**
- The test set adds France (15% of Source-1), with generic names ("Ecole SAS", "Lycée Reseau") and French abbreviations (R., Av, Bd, Pl, BIS).
- Because of this, every learned weight is data-derived per split, and no feature one-hot encodes the country.

### 2.2 Solution Strategy

**Approach Type:** Blocking → cheap re-ranker → pair classifier (LightGBM) → assignment + expected-F0.5 subset selection

**Core Innovation:** Blocking and assignment run from the query side, using the structural constraint that each S2/S3 record matches at most one entity. On top of that sit a two-stage learned cascade and an explicit per-entity expected-F0.5 decision, which can predict an empty list for likely singletons.

---

## 3. Candidate Generation (Blocking)

**Normalization** is applied to every record before key generation:
- Unicode folding and a single ISCII-offset transliterator for all 9 Brahmic blocks (inherent-vowel handling and schwa deletion).
- A lexicon learned from positionally aligned training pairs: 1,347 Indic tokens (for example प्राइवेट → private) and 16 native-script state names.
- Legal-form canonicalization (Private/Pvt → pvt, S.A.S. → sas, Cie → co) and single-letter merging.
- Leet-speak repair.
- US/Indian state canonicalization, applied only to a whole address component.
- Street-type abbreviations in English and French, ordinal and number-word normalization, and leading-zero stripping.
- Phone and domain extraction.
- A consonant-skeleton key for spelling-tolerant matching.

**Blocking keys used** (hashed, scoped by country, IDF-weighted):

| Key family | Examples / purpose |
|---|---|
| Name tokens (T) | Single tokens of the name core. |
| Name token pairs (P) and triples (R) | Pairs/triples of name-core tokens; specific enough when single words are common. |
| Name skeletons (S) and skeleton pairs (K) | Robust to typos and transliteration. |
| Concatenated name (C) and 10-char prefix/suffix (X) | Catch domain-style names. |
| Address number × address word (A) | Also used to catch DBA names. |
| Name word × address word (N) | |
| Name word × house number (M) | |
| Address word pairs (W) and address number pairs (U) | |
| Phone numbers (Q) | |

- Keys whose Source-1 document frequency exceeds a cap (60 for single tokens, 200 for combinations) are dropped.
- Each S1 candidate is scored by the sum of log(N/df) over shared keys, and the top 30 per query are kept.
- **Cheap re-ranker:** a LightGBM model (29 vectorized features: blocking score, rank and gap, key-family bitmask, and rapidfuzz name/address similarities) keeps at most 6 candidates per query with probability ≥ 0.002.

**Candidate pairs generated (train):**

| Stage | Pairs | Recall of all true pairs |
|---|---|---|
| Blocking, top 30 per query | 290.3M | 98.94% |
| After re-ranker (≈1.2 per query) | 12.5M | **98.56%** |

**Test:** see `output/candidate_pairs.tsv`.

**How we kept true matches:**
- Several independent key families mean a typo in one field doesn't lose the pair.
- Address-only keys cover DBA names, and name-only keys cover missing addresses.
- Skeleton keys cover transliteration.
- The re-ranker's floor is set very low, so it only removes candidates that are clearly impossible.

---

## 4. Matching Model

**Features used (~70 per pair):**

**Name features**
- rapidfuzz ratio, token-sort, token-set and partial ratio on the normalized name and its core.
- Jaro-Winkler and ratio on the concatenated name (for domains), and partial ratio for containment.
- Skeleton ratio and token-set ratio.
- IDF-weighted exact and fuzzy token recall in both directions (fuzzy threshold: Jaro-Winkler ≥ 0.84).
- IDF of the rarest shared token and of the rarest *unmatched* token on each side, and the total name specificity.
- Legal-form agreement: same / missing / compatible / conflicting.
- Token counts, plus flags for domain names and Indic script.

**Address features**
- Token-set ratio, ratio, and partial token-set ratio.
- IDF-weighted exact and fuzzy overlap of address words.
- House-number overlap: count, recall, Jaccard, first number equal, first number contained.
- State agreement, and missing-address flags.

**Other features**
- Phone agreement and source id.
- Blocking score, rank, relative score and gap to the second candidate, number of shared keys, and a 13-bit key-family mask.
- The re-ranker probability and its rank and gap within the query.
- Source-1-side competition: how many queries point at the entity, this query's rank among them, and the sum of re-ranker probabilities.

**Model type:**
- LightGBM binary classifier (255 leaves, learning rate 0.1, up to 300 rounds with early stopping), trained on a 3M-pair sample.
- 2-fold cross-fitting with folds split by Source-1 entity (hash), so no entity appears in both training and scoring.
- Test predictions average the two fold models.

**Threshold selection method:** chosen by grid search on out-of-fold macro F0.5 over all 2.2M training entities, comparing:
1. Assignment (each S2/S3 record keeps only its highest-scoring S1 candidate), then a probability threshold.
2. Assignment, then per-entity subset selection maximizing expected F0.5, E[F] ≈ 1.25·Σp_top-k / (k + 0.25·Σp). This is compared against the probability that the entity is a singleton, Π(1−p), so the empty list is predicted when that is the better bet.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, out-of-fold):** **0.9798**, measured on all 883,188 India training entities.
  India records link only to India entities, so this subset is a complete, exact evaluation. The full-train evaluation was skipped because of time and RAM limits.
- Micro precision: 0.9942 / micro recall: 0.9533 / singleton accuracy: 0.9694
- Recall ceiling from candidate generation: 98.56%

- **Common false positives (wrong merges):**
  - Generic names ("Summit Inc", "Global Center") with a missing address, where a distractor record shares the only informative token.
  - Chains or franchises: the same name in the same city with different house numbers.
- **Common false negatives (missed matches):**
  - Records combining a heavily corrupted name ("Sthffawte" for Software) with a missing address.
  - DBA names whose address shares no number with the reference record.
  - Indic-script words missing from the learned lexicon whose rule-based transliteration differs a lot from the English spelling.

---

## 6. Conclusion

A carefully engineered normalization layer (transliteration, canonical legal forms, number and state handling) combined with multi-key IDF blocking reaches a 98.6% recall ceiling at only 1.2 candidates per record. The precision-oriented metric is handled by structural constraints (at most one S1 per record) and an explicit expected-F0.5 decision, not a single global threshold. Because all IDF statistics are recomputed per split and every feature is country-agnostic, the pipeline carries over to the unseen France data without code changes.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (Python 3.12, CPU only; see its `README.md`):

```
src/ber/config.py      hyper-parameters and paths
src/ber/translit.py    Brahmic transliteration + learned lexicon
src/ber/normalize.py   name/address normalization
src/ber/prepare.py     loading, lexicon learning, parallel normalization
src/ber/blocking.py    13 blocking-key families, IDF candidate scoring
src/ber/features.py    cheap + full pair features
src/ber/model.py       LightGBM helpers
src/ber/decide.py      assignment + expected-F0.5 selection
src/ber/evaluate.py    macro F0.5 metric
src/ber/pipeline.py    CLI entry point
```

To reproduce both output files from `src/`, run:

```
python -m ber.pipeline all --data-dir <dataset> --work-dir <work> --output-dir <output>
```

This runs, in order: prepare → rerank-train → candidates (train, test) → train → predict.

### B. Additional Results

Decision-rule grid (out-of-fold macro F0.5, India):

| Rule | Macro F0.5 | Precision | Recall | Singleton acc. |
|---|---|---|---|---|
| Assign + threshold 0.30 | 0.9735 | 0.9807 | 0.9686 | 0.9425 |
| Assign + threshold 0.50 | 0.9781 | 0.9884 | 0.9630 | 0.9660 |
| Assign + threshold 0.70 | 0.9795 | 0.9935 | 0.9552 | 0.9806 |
| Assign + threshold 0.90 | 0.9759 | 0.9972 | 0.9381 | 0.9919 |
| **Assign + expected-F0.5 (floor 0.5, empty bias 1.3), chosen** | **0.9798** | 0.9942 | 0.9533 | 0.9694 |

**Deadline note:** a stacked second-stage "context" model (`use_context_model` in `config.py`) is implemented but was disabled for this submission. It needs more RAM and time than were available.
