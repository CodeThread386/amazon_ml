"""Pair features.

Two tiers:
  * cheap_features  - fully vectorized (rapidfuzz cpdist + blocking evidence); used by the
                      re-ranker that prunes ~30 blocking candidates per query to a handful.
  * full_features   - adds IDF-weighted exact/fuzzy token overlap for names and addresses,
                      number/phone/legal-form/state agreement and query/S1 competition context.

All similarity features are language-agnostic and IDF tables are computed from the split
being scored, so unseen countries get sensible weights without retraining.
"""
import math
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from .blocking import KEY_TYPES
from .config import N_JOBS

REC_COLS = ["rid", "src", "country", "name_n", "name_core", "name_cat", "name_full_cat", "name_skel",
            "legal", "is_domain", "is_indic", "name_phones", "addr_n", "addr_nums", "addr_alpha",
            "state", "addr_phones", "addr_missing", "addr_first_num"]


def attach_records(cand: pl.DataFrame, records: pl.DataFrame) -> pl.DataFrame:
    """Joins S1 (suffix _s) and query (suffix _q) record fields onto candidate pairs."""
    r = records.select([c for c in REC_COLS if c in records.columns])
    s = r.rename({c: f"{c}_s" for c in r.columns if c != "rid"}).rename({"rid": "s1_rid"})
    q = r.rename({c: f"{c}_q" for c in r.columns if c != "rid"}).rename({"rid": "q_rid"})
    return cand.join(s, on="s1_rid", how="left").join(q, on="q_rid", how="left")


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _sim_block(p: pl.DataFrame) -> dict:
    core_q, core_s = p["name_core_q"].to_list(), p["name_core_s"].to_list()
    addr_q, addr_s = p["addr_n_q"].to_list(), p["addr_n_s"].to_list()
    cat_q, cat_s = p["name_cat_q"].to_list(), p["name_cat_s"].to_list()
    skel_q, skel_s = p["name_skel_q"].to_list(), p["name_skel_s"].to_list()
    return {
        "nm_ratio": _cp(core_q, core_s, fuzz.ratio),
        "nm_tset_full": _cp(p["name_n_q"].to_list(), p["name_n_s"].to_list(), fuzz.token_set_ratio),
        "nm_cat_jw": _cp(cat_q, cat_s, JaroWinkler.normalized_similarity),
        "nm_skel_tset": _cp(skel_q, skel_s, fuzz.token_set_ratio),
        "ad_tset": _cp(addr_q, addr_s, fuzz.token_set_ratio),
        "_core_q": core_q, "_core_s": core_s, "_addr_q": addr_q, "_addr_s": addr_s,
        "_cat_q": cat_q, "_cat_s": cat_s, "_skel_q": skel_q, "_skel_s": skel_s,
    }


def _block_cols(p: pl.DataFrame) -> list:
    cols = [
        pl.col("bscore"), pl.col("brank").cast(pl.Float32),
        (pl.col("bscore") / pl.col("bmax")).alias("brel"),
        pl.col("nkeys").cast(pl.Float32),
        pl.len().over("q_rid").cast(pl.Float32).alias("q_ncand"),
        (pl.col("bscore") - pl.col("bscore").filter(pl.col("brank") == 2).first().over("q_rid"))
        .fill_null(pl.col("bscore")).alias("bgap2"),
        pl.col("addr_missing_q").cast(pl.Int8).alias("q_addr_missing"),
        pl.col("addr_missing_s").cast(pl.Int8).alias("s_addr_missing"),
        pl.col("is_domain_q").cast(pl.Int8).alias("q_is_domain"),
        pl.col("is_indic_q").cast(pl.Int8).alias("q_is_indic"),
        pl.col("src_q").cast(pl.Int8).alias("q_src"),
    ]
    for name, bit in KEY_TYPES.items():
        cols.append(((pl.col("ktmask").cast(pl.Int32) // (1 << bit)) % 2).cast(pl.Int8).alias(f"kt_{name}"))
    return cols


def cheap_features(p: pl.DataFrame) -> pl.DataFrame:
    sims = _sim_block(p)
    out = p.select("q_rid", "s1_rid", *_block_cols(p))
    return out.with_columns([pl.Series(k, v) for k, v in sims.items() if not k.startswith("_")])


# ----------------------------------------------------------------------------- per-pair python features

_IDF = None


def _init_idf(idf):
    global _IDF
    _IDF = idf


def _weights(tokens, table, default):
    return [table.get(t, default) for t in tokens]


def _overlap(ts, tq, table, default):
    """IDF-weighted exact and fuzzy overlap between S1 tokens ts and query tokens tq."""
    if not ts or not tq:
        return (-1.0,) * 8
    ws, wq = _weights(ts, table, default), _weights(tq, table, default)
    sws, swq = sum(ws), sum(wq)
    setq, sets = set(tq), set(ts)
    common_w = 0.0
    max_common = 0.0
    for t, w in zip(ts, ws):
        if t in setq:
            common_w += w
            max_common = max(max_common, w)
    best_q = [1.0 if u in sets else 0.0 for u in tq]
    fuzzy_s = 0.0
    max_unmatched_s = 0.0
    jw = JaroWinkler.normalized_similarity
    for t, w in zip(ts, ws):
        if t in setq:
            fuzzy_s += w
            continue
        best = 0.0
        for j, u in enumerate(tq):
            s = jw(t, u)
            if s > best:
                best = s
            if s > best_q[j]:
                best_q[j] = s
        if best >= 0.84:
            fuzzy_s += w * best
        elif w > max_unmatched_s:
            max_unmatched_s = w
    fuzzy_q = 0.0
    max_unmatched_q = 0.0
    for b, w in zip(best_q, wq):
        if b >= 0.84:
            fuzzy_q += w * b
        elif w > max_unmatched_q:
            max_unmatched_q = w
    return (common_w / sws, common_w / swq, fuzzy_s / sws, fuzzy_q / swq, max_common,
            max_unmatched_s, max_unmatched_q, sws)


def _pair_block(args):
    (country_s, core_s, core_q, alpha_s, alpha_q, nums_s, nums_q, first_s, first_q,
     phones_s, phones_q, legal_s, legal_q, state_s, state_q) = args
    rows = []
    for i in range(len(core_s)):
        c = country_s[i]
        ntab, atab, ndef, adef = _IDF[c]
        ts = [t for t in core_s[i].split(" ") if t]
        tq = [t for t in core_q[i].split(" ") if t]
        nf = _overlap(ts, tq, ntab, ndef)
        af = _overlap(alpha_s[i] or [], alpha_q[i] or [], atab, adef)
        ns, nq = set(nums_s[i] or []), set(nums_q[i] or [])
        if ns and nq:
            inter = len(ns & nq)
            num = (inter, inter / len(nq), inter / len(ns | nq), float(first_s[i] == first_q[i]),
                   float(first_q[i] in ns))
        else:
            num = (-1.0, -1.0, -1.0, -1.0, -1.0)
        ps, pq = set(phones_s[i] or []), set(phones_q[i] or [])
        phone = float(bool(ps & pq)) if (ps and pq) else -1.0
        ls, lq = set(legal_s[i].split()) if legal_s[i] else set(), set(legal_q[i].split()) if legal_q[i] else set()
        if not ls and not lq:
            legal = 1.0
        elif ls == lq:
            legal = 2.0
        elif not ls or not lq:
            legal = 3.0
        elif ls & lq:
            legal = 4.0
        else:
            legal = 0.0
        st = -1.0 if (not state_s[i] or not state_q[i]) else float(state_s[i] == state_q[i])
        rows.append(nf + af + num + (phone, legal, st, float(len(ts)), float(len(tq))))
    return np.asarray(rows, dtype=np.float32)


PAIR_COLS = (
    ["nm_w_rec_s", "nm_w_rec_q", "nm_fz_rec_s", "nm_fz_rec_q", "nm_max_common", "nm_max_unm_s",
     "nm_max_unm_q", "nm_idf_sum_s"]
    + ["ad_w_rec_s", "ad_w_rec_q", "ad_fz_rec_s", "ad_fz_rec_q", "ad_max_common", "ad_max_unm_s",
       "ad_max_unm_q", "ad_idf_sum_s"]
    + ["num_common", "num_rec_q", "num_jac", "num_first_eq", "num_firstq_in_s"]
    + ["phone_match", "legal_rel", "state_match", "nm_ntok_s", "nm_ntok_q"]
)


def build_idf(records: pl.DataFrame) -> dict:
    """country -> (name_idf, addr_idf, name_default, addr_default), from all sources of the split."""
    out = {}
    for c in records["country"].unique().to_list():
        r = records.filter(pl.col("country") == c)
        n = r.height
        nt = (r.select(pl.col("name_core").str.split(" ").list.unique().alias("t")).explode("t")
                .group_by("t").len())
        at = (r.select(pl.col("addr_alpha").list.unique().alias("t")).explode("t").drop_nulls()
                .group_by("t").len())
        ntab = dict(zip(nt["t"].to_list(), (math.log(n / v) for v in nt["len"].to_list())))
        atab = dict(zip(at["t"].to_list(), (math.log(n / v) for v in at["len"].to_list())))
        out[c] = (ntab, atab, math.log(n), math.log(n))
    return out


class PairFeaturizer:
    """Holds a worker pool with the IDF tables loaded once per worker."""

    def __init__(self, idf, n_jobs=N_JOBS):
        self.pool = Pool(n_jobs, initializer=_init_idf, initargs=(idf,))

    def close(self):
        self.pool.close()
        self.pool.join()

    def pair_features(self, p: pl.DataFrame, block=20_000) -> np.ndarray:
        cols = [p[c].to_list() for c in (
            "country_s", "name_core_s", "name_core_q", "addr_alpha_s", "addr_alpha_q", "addr_nums_s",
            "addr_nums_q", "addr_first_num_s", "addr_first_num_q", "name_phones_s", "name_phones_q",
            "legal_s", "legal_q", "state_s", "state_q")]
        # phones may appear in either field
        ph_s = [(a or []) + (b or []) for a, b in zip(cols[9], p["addr_phones_s"].to_list())]
        ph_q = [(a or []) + (b or []) for a, b in zip(cols[10], p["addr_phones_q"].to_list())]
        cols[9], cols[10] = ph_s, ph_q
        n = p.height
        tasks = [tuple(c[i:i + block] for c in cols) for i in range(0, n, block)]
        parts = self.pool.map(_pair_block, tasks)
        return np.vstack(parts) if parts else np.zeros((0, len(PAIR_COLS)), np.float32)


def full_features(p: pl.DataFrame, featurizer: PairFeaturizer) -> pl.DataFrame:
    sims = _sim_block(p)
    extra = {
        "nm_tsort": _cp(sims["_core_q"], sims["_core_s"], fuzz.token_sort_ratio),
        "nm_partial": _cp(sims["_core_q"], sims["_core_s"], fuzz.partial_ratio),
        "nm_fullcat_ratio": _cp(p["name_full_cat_q"].to_list(), p["name_full_cat_s"].to_list(), fuzz.ratio),
        "nm_cat_partial": _cp(sims["_cat_q"], sims["_cat_s"], fuzz.partial_ratio),
        "nm_skel_ratio": _cp(sims["_skel_q"], sims["_skel_s"], fuzz.ratio),
        "ad_ratio": _cp(sims["_addr_q"], sims["_addr_s"], fuzz.ratio),
        "ad_partial": _cp(sims["_addr_q"], sims["_addr_s"], fuzz.partial_token_set_ratio),
    }
    pf = featurizer.pair_features(p)
    out = p.select("q_rid", "s1_rid", *_block_cols(p), "cheap", "cheap_rank", "cheap_gap")
    return out.with_columns([pl.Series(k, v) for k, v in sims.items() if not k.startswith("_")]
                            + [pl.Series(k, v) for k, v in extra.items()]
                            + [pl.Series(c, pf[:, i]) for i, c in enumerate(PAIR_COLS)])


def s1_context(cand: pl.DataFrame, score: str) -> pl.DataFrame:
    """Competition among queries that point at the same Source-1 record (needs the full graph)."""
    return cand.select("q_rid", "s1_rid",
                       pl.len().over("s1_rid").cast(pl.Float32).alias("s1_degree"),
                       pl.col(score).rank("ordinal", descending=True).over("s1_rid")
                         .cast(pl.Float32).alias(f"s1_{score}_rank"),
                       pl.col(score).sum().over("s1_rid").alias(f"s1_{score}_sum"),
                       (pl.col(score) > 0.5).sum().over("s1_rid").cast(pl.Float32).alias(f"s1_{score}_n50"))
