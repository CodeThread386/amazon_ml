"""End-to-end pipeline.

    python -m ber.pipeline all            # everything below, in order
    python -m ber.pipeline prepare        # lexicon + normalization (train, test)
    python -m ber.pipeline rerank-train   # train the cheap re-ranker on a query sample
    python -m ber.pipeline candidates --split train|test
    python -m ber.pipeline train          # cross-fitted pair model + context model + decision tuning
    python -m ber.pipeline predict        # score test and write output/*.tsv
"""
import argparse
import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import decide, model
from .blocking import KEY_COLS, build_index, query_candidates
from .config import BlockingParams, ModelParams, Paths
from .evaluate import macro_f05
from .features import (REC_COLS, PairFeaturizer, attach_records, build_idf, cheap_features,
                       full_features, s1_context)
from .prepare import (countries, learn_lexicon, load_records, prepare_split, read_ground_truth)

log = logging.getLogger("ber")

REC_ALL = sorted(set(REC_COLS) | set(KEY_COLS))

CTX_RAW = ["nm_w_rec_s", "nm_w_rec_q", "ad_w_rec_s", "ad_w_rec_q", "nm_tset_full", "ad_tset",
           "num_common", "num_first_eq", "q_addr_missing", "s_addr_missing", "q_src", "q_is_indic",
           "q_is_domain", "nm_idf_sum_s", "brel", "cheap"]


# ----------------------------------------------------------------------------- helpers

def _labels(paths: Paths) -> pl.DataFrame:
    ids = load_records(paths, "train", columns=["rid", "entity_id"])
    gt = read_ground_truth(paths)
    return (gt.join(ids.rename({"rid": "s1_rid", "entity_id": "s1_id"}), on="s1_id")
              .join(ids.rename({"rid": "q_rid", "entity_id": "q_id"}), on="q_id")
              .select("q_rid", "s1_rid", pl.lit(1, dtype=pl.UInt8).alias("label")))


def _with_label(df: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    return df.join(labels, on=["q_rid", "s1_rid"], how="left").with_columns(
        pl.col("label").fill_null(0).cast(pl.UInt8))


def _query_rank_cols(score: str) -> list:
    rank = pl.col(score).rank("ordinal", descending=True).over("q_rid")
    best = pl.col(score).max().over("q_rid")
    second = pl.col(score).filter(rank == 2).first().over("q_rid").fill_null(0.0)
    return [
        rank.cast(pl.Float32).alias(f"{score}_rank"),
        pl.when(rank == 1).then(pl.col(score) - second).otherwise(pl.col(score) - best)
          .cast(pl.Float32).alias(f"{score}_gap"),
    ]


def _chunks(df: pl.DataFrame, size: int):
    for i in range(0, df.height, size):
        yield i // size, df.slice(i, size)


# ----------------------------------------------------------------------------- steps

def step_prepare(paths: Paths):
    lex = learn_lexicon(paths)
    lex.save(paths.work("lexicon.json"))
    for split in ("train", "test"):
        prepare_split(paths, split, paths.work("lexicon.json"))


def step_rerank_train(paths: Paths, bp: BlockingParams):
    labels = _labels(paths)
    rows = []
    for country in countries(paths, "train"):
        recs = load_records(paths, "train", columns=REC_ALL, country=country)
        s1 = recs.filter(pl.col("src") == 1)
        q = recs.filter(pl.col("src") != 1)
        q = q.sample(min(bp.rerank_sample_per_country, q.height), seed=11)
        index = build_index(s1, bp)
        for _, qc in _chunks(q, bp.query_chunk):
            c = query_candidates(qc, index, bp)
            rows.append(_with_label(cheap_features(attach_records(c, recs)), labels))
        del index, recs
    df = pl.concat(rows).with_columns((pl.col("q_rid").hash(seed=5) % 10 == 0).alias("_va"))
    tr, va = df.filter(~pl.col("_va")).drop("_va"), df.filter(pl.col("_va")).drop("_va")
    log.info("re-ranker data: %d rows, %.3f positive", df.height, df["label"].mean())
    params = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 100,
              "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": 13, "verbose": -1}
    booster = model.train_lgb(tr, params, 400, valid=va)
    log.info("re-ranker importance: %s", model.importance(booster, 12))
    booster.save_model(str(paths.work("models", "rerank.lgb")))


def step_candidates(paths: Paths, split: str, bp: BlockingParams):
    t0 = time.time()
    rerank = lgb.Booster(model_file=str(paths.work("models", "rerank.lgb")))
    labels = _labels(paths) if split == "train" else None
    idf = build_idf(load_records(paths, split, columns=["country", "name_core", "addr_alpha"]))
    featurizer = PairFeaturizer(idf)
    del idf
    out_dir = paths.work(split, "feat", "x").parent
    out_dir.mkdir(parents=True, exist_ok=True)
    done_countries = {f.name.split("_")[0] for f in out_dir.glob("*.parquet")}
    stats = []
    for country in countries(paths, split):
        if country in done_countries and labels is None:
            log.info("[%s/%s] already done, skipping", split, country)
            continue
        recs = load_records(paths, split, columns=REC_ALL, country=country)
        s1 = recs.filter(pl.col("src") == 1)
        q = recs.filter(pl.col("src") != 1)
        index = build_index(s1, bp)
        log.info("[%s/%s] s1=%d queries=%d postings=%d", split, country, s1.height, q.height, index.height)
        for j, qc in _chunks(q, bp.query_chunk):
            c = query_candidates(qc, index, bp)
            a = attach_records(c, recs)
            a = a.with_columns(pl.Series("cheap", model.predict(rerank, cheap_features(a))))
            a = a.with_columns(_query_rank_cols("cheap"))
            keep = a.filter((pl.col("cheap_rank") <= bp.keep_k) & (pl.col("cheap") >= bp.rerank_floor))
            ff = full_features(keep, featurizer)
            if labels is not None:
                ff = _with_label(ff, labels)
                wide_tp = c.join(labels, on=["q_rid", "s1_rid"]).height
                stats.append((country, wide_tp, int(ff["label"].sum()), c.height, ff.height))
            ff.write_parquet(out_dir / f"{country}_{j:05d}.parquet")
            if j % 20 == 0:
                log.info("[%s/%s] chunk %d/%d, %d wide -> %d kept pairs (%.0fs)", split, country, j,
                         (q.height - 1) // bp.query_chunk, c.height, ff.height, time.time() - t0)
        del index, recs
        log.info("[%s/%s] done (%.0fs)", split, country, time.time() - t0)
    featurizer.close()
    files = sorted(out_dir.glob("*.parquet"))
    ctx = pl.scan_parquet(files).select("q_rid", "s1_rid", "cheap").collect()
    s1_context(ctx, "cheap").write_parquet(paths.work(split, "s1ctx.parquet"))
    report = {"pairs_kept": ctx.height, "queries": ctx["q_rid"].n_unique()}
    if labels is not None:
        total = labels.height
        report.update({
            "true_pairs": total,
            "recall_wide": sum(s[1] for s in stats) / total,
            "recall_kept": sum(s[2] for s in stats) / total,
            "pairs_wide": sum(s[3] for s in stats),
        })
    log.info("candidates[%s]: %s", split, report)
    with open(paths.work(split, "candidates_report.json"), "w") as f:
        json.dump(report, f, indent=2)


def _load_features(paths: Paths, split: str, frac: float = 1.0, seed: int = 0, n_folds: int = 2):
    ctx = pl.read_parquet(paths.work(split, "s1ctx.parquet"))
    for f in sorted(paths.work(split, "feat", "x").parent.glob("*.parquet")):
        df = pl.read_parquet(f)
        if frac < 1.0:
            df = df.filter((pl.col("q_rid").hash(seed=seed) % 1000) < int(frac * 1000))
        df = df.join(ctx, on=["q_rid", "s1_rid"], how="left")
        yield df.with_columns(model.fold_of(pl.col("s1_rid"), n_folds).alias("fold"))


def _ctx_features(oof: pl.DataFrame) -> pl.DataFrame:
    """Second-order features from the pair model's probabilities over the whole candidate graph."""
    df = oof.with_columns(_query_rank_cols("p"))
    rank_s1 = pl.col("p").rank("ordinal", descending=True).over("s1_rid")
    best_s1 = pl.col("p").max().over("s1_rid")
    second_s1 = pl.col("p").filter(rank_s1 == 2).first().over("s1_rid").fill_null(0.0)
    is_q_best = pl.col("p_rank") == 1
    return df.with_columns(
        pl.col("p").sum().over("q_rid").alias("q_psum"),
        (pl.col("p") > 0.1).sum().over("q_rid").cast(pl.Float32).alias("q_n10"),
        rank_s1.cast(pl.Float32).alias("p_rank_s1"),
        pl.when(rank_s1 == 1).then(second_s1).otherwise(best_s1).alias("s1_p_other_best"),
        pl.col("p").sum().over("s1_rid").alias("s1_psum"),
        pl.col("p").filter(is_q_best).sum().over("s1_rid").fill_null(0.0).alias("s1_psum_assigned"),
        (pl.col("p") > 0.5).sum().over("s1_rid").cast(pl.Float32).alias("s1_n50"),
        pl.len().over("s1_rid").cast(pl.Float32).alias("s1_ncand"),
    )


CTX_FEATURES = ["p", "p_rank", "p_gap", "q_psum", "q_n10", "p_rank_s1", "s1_p_other_best", "s1_psum",
                "s1_psum_assigned", "s1_n50", "s1_ncand", "s1_degree"] + CTX_RAW


def _cross_fit(sample: pl.DataFrame, features, params, rounds, n_folds, seed):
    boosters = []
    for k in range(n_folds):
        tr = sample.filter(pl.col("fold") == k)
        va_mask = (pl.col("q_rid").hash(seed=seed) % 20) == 0
        booster = model.train_lgb(tr.filter(~va_mask), params, rounds, valid=tr.filter(va_mask),
                                  features=features)
        log.info("fold %d importance: %s", k, model.importance(booster, 15))
        boosters.append(booster)
    return boosters


def _oof(boosters, df: pl.DataFrame) -> np.ndarray:
    """Out-of-fold prediction: rows of fold k are scored by the model trained on another fold."""
    out = np.zeros(df.height, dtype=np.float32)
    folds = df["fold"].to_numpy()
    n = len(boosters)
    for k in range(n):
        idx = np.flatnonzero(folds == k)
        if len(idx):
            out[idx] = model.predict(boosters[(k + 1) % n], df[idx])
    return out


def _evaluate_decisions(scored: pl.DataFrame, score: str, truth: pl.DataFrame, s1_ids: pl.Series):
    """Grid over decision rules on out-of-fold scores; everything stays in integer-id space."""
    assigned = decide.assign(scored.select("q_rid", "s1_rid", score), score)

    def score_pairs(pairs):
        return macro_f05(pairs.rename({"s1_rid": "s1_id", "q_rid": "q_id"}), truth, s1_ids)

    results = []
    for t in np.arange(0.3, 0.91, 0.05):
        r = score_pairs(decide.select_threshold(assigned, float(t), score))
        results.append(({"mode": "threshold", "t": round(float(t), 3)}, r))
        log.info("[%s] threshold %.2f -> %s", score, t, {k: round(v, 5) for k, v in r.items() if k != "n_entities"})
    best_t = max(results, key=lambda x: x[1]["macro_f05"])
    base_floor = max(0.1, best_t[0]["t"] - 0.2)
    for floor in (base_floor, best_t[0]["t"] - 0.1):
        for bias in (1.0, 1.3):
            cfg = {"mode": "expected_f", "floor": round(floor, 3), "missing": 0.0, "empty_bias": bias}
            r = score_pairs(decide.select_expected_f(assigned, floor, 0.0, bias, score))
            results.append((cfg, r))
            log.info("[%s] %s -> %s", score, cfg, {k: round(v, 5) for k, v in r.items() if k != "n_entities"})
    best = max(results, key=lambda x: x[1]["macro_f05"])
    log.info("[%s] best threshold %s ; best overall %s", score, best_t, best)
    return best


def step_train(paths: Paths, mp: ModelParams):
    files = sorted(paths.work("train", "feat", "x").parent.glob("*.parquet"))
    total = pl.scan_parquet(files).select(pl.len()).collect().item()
    frac = min(1.0, mp.train_pairs_per_fold * mp.n_folds / total)
    log.info("train pairs %d, sampling %.3f for the pair model", total, frac)
    sample = pl.concat(list(_load_features(paths, "train", frac, seed=mp.seed, n_folds=mp.n_folds)))
    features = model.feature_names(sample)
    log.info("pair model: %d rows, %d features, %.4f positive", sample.height, len(features), sample["label"].mean())
    boosters = _cross_fit(sample, features, mp.lgb, mp.num_boost_round, mp.n_folds, mp.seed)
    del sample
    for k, b in enumerate(boosters):
        b.save_model(str(paths.work("models", f"pair_fold{k}.lgb")))

    keep_cols = ["q_rid", "s1_rid", "label", "fold"] + (["s1_degree", *CTX_RAW] if mp.use_context_model else [])
    oof = pl.concat([df.select(keep_cols).with_columns(pl.Series("p", _oof(boosters, df)))
                     for df in _load_features(paths, "train", n_folds=mp.n_folds)])
    truth = _labels(paths).select(pl.col("s1_rid").alias("s1_id"), pl.col("q_rid").alias("q_id"))
    s1_ids = load_records(paths, "train", srcs=(1,), columns=["rid"])["rid"].alias("s1_id")
    best_p = _evaluate_decisions(oof, "p", truth, s1_ids)
    summary = {"decision": {"score": "p", **best_p[0]}, "pair_model_oof": best_p[1]}

    if mp.use_context_model:
        oof = _ctx_features(oof)
        ctx_params = {**mp.lgb, "num_leaves": 63, "learning_rate": 0.05, "min_data_in_leaf": 500}
        ctx_frac = min(1.0, 8_000_000 / oof.height)
        ctx_sample = oof.filter((pl.col("q_rid").hash(seed=mp.seed + 1) % 1000) < int(ctx_frac * 1000))
        ctx_boosters = _cross_fit(ctx_sample, CTX_FEATURES, ctx_params, 800, mp.n_folds, mp.seed + 1)
        del ctx_sample
        for k, b in enumerate(ctx_boosters):
            b.save_model(str(paths.work("models", f"ctx_fold{k}.lgb")))
        oof = oof.with_columns(pl.Series("p2", _oof(ctx_boosters, oof)))
        best_p2 = _evaluate_decisions(oof, "p2", truth, s1_ids)
        summary["context_model_oof"] = best_p2[1]
        if best_p2[1]["macro_f05"] > best_p[1]["macro_f05"]:
            summary["decision"] = {"score": "p2", **best_p2[0]}

    oof.select("q_rid", "s1_rid", "label", *[c for c in ("p", "p2") if c in oof.columns]).write_parquet(
        paths.work("train", "oof.parquet"))
    log.info("validation summary: %s", summary)
    with open(paths.work("models", "decision.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _avg_predict(boosters, df: pl.DataFrame) -> np.ndarray:
    return np.mean([model.predict(b, df) for b in boosters], axis=0).astype(np.float32)


def step_predict(paths: Paths, mp: ModelParams):
    with open(paths.work("models", "decision.json")) as f:
        decision = json.load(f)["decision"]
    pair = [lgb.Booster(model_file=str(paths.work("models", f"pair_fold{k}.lgb"))) for k in range(mp.n_folds)]
    score = decision["score"]
    cols = ["q_rid", "s1_rid"] + (["s1_degree", *CTX_RAW] if score == "p2" else [])
    scored = pl.concat([df.select(cols).with_columns(pl.Series("p", _avg_predict(pair, df)))
                        for df in _load_features(paths, "test", n_folds=mp.n_folds)])
    if score == "p2":
        ctxm = [lgb.Booster(model_file=str(paths.work("models", f"ctx_fold{k}.lgb"))) for k in range(mp.n_folds)]
        scored = _ctx_features(scored)
        scored = scored.with_columns(pl.Series("p2", _avg_predict(ctxm, scored)))
    assigned = decide.assign(scored.select("q_rid", "s1_rid", score), score)
    if decision["mode"] == "threshold":
        chosen = decide.select_threshold(assigned, decision["t"], score)
    else:
        chosen = decide.select_expected_f(assigned, decision["floor"], decision["missing"],
                                          decision["empty_bias"], score)
    write_outputs(paths, scored.select("q_rid", "s1_rid"), chosen)


def _id_lists(paths: Paths, pairs: pl.DataFrame, col: str) -> pl.DataFrame:
    rid2id = load_records(paths, "test", columns=["rid", "entity_id", "src"])
    s1 = rid2id.filter(pl.col("src") == 1).select(pl.col("rid").alias("s1_rid"),
                                                   pl.col("entity_id").alias("source1_entity_id"))
    q = rid2id.filter(pl.col("src") != 1).select(pl.col("rid").alias("q_rid"), pl.col("entity_id").alias("qid"))
    lists = (pairs.join(q, on="q_rid").sort("qid").group_by("s1_rid")
                  .agg(pl.col("qid").unique(maintain_order=True).str.join(",").alias(col)))
    return (s1.join(lists, on="s1_rid", how="left").with_columns(pl.col(col).fill_null(""))
              .select("source1_entity_id", col))


def write_outputs(paths: Paths, candidates: pl.DataFrame, chosen: pl.DataFrame):
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    m = _id_lists(paths, chosen, "matched_entity_ids")
    c = _id_lists(paths, candidates, "candidate_entity_ids")
    m.write_csv(paths.output_dir / "matching_results.tsv", separator="\t", quote_style="never")
    c.write_csv(paths.output_dir / "candidate_pairs.tsv", separator="\t", quote_style="never")
    n_matched = (m["matched_entity_ids"] != "").sum()
    log.info("wrote %d entities (%d with matches, %d matched ids, %d candidate pairs)",
             m.height, n_matched, chosen.height, candidates.height)


# ----------------------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description="Business entity resolution pipeline")
    ap.add_argument("step", choices=["all", "prepare", "rerank-train", "candidates", "train", "predict"])
    ap.add_argument("--split", choices=["train", "test"], default=None)
    root = Path(__file__).resolve().parents[4]
    ap.add_argument("--data-dir", type=Path, default=root / "dataset")
    ap.add_argument("--work-dir", type=Path, default=root / "work")
    ap.add_argument("--output-dir", type=Path, default=root / "output")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    paths = Paths(args.data_dir, args.work_dir, args.output_dir)
    bp, mp = BlockingParams(), ModelParams()
    steps = {
        "prepare": lambda: step_prepare(paths),
        "rerank-train": lambda: step_rerank_train(paths, bp),
        "candidates": lambda: [step_candidates(paths, s, bp) for s in ([args.split] if args.split else ["train", "test"])],
        "train": lambda: step_train(paths, mp),
        "predict": lambda: step_predict(paths, mp),
    }
    order = ["prepare", "rerank-train", "candidates", "train", "predict"] if args.step == "all" else [args.step]
    for s in order:
        t = time.time()
        log.info("=== %s ===", s)
        steps[s]()
        log.info("=== %s finished in %.0fs ===", s, time.time() - t)


if __name__ == "__main__":
    main()
