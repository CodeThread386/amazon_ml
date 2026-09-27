"""Turning pair probabilities into the final match lists.

1. Assignment: an S2/S3 record belongs to at most one Source-1 entity, so it keeps only its
   highest-probability candidate.
2. Per-entity subset selection maximizing expected F_0.5. For an entity with assigned
   probabilities p_1 >= p_2 >= ..., predicting the top k gives approximately
       E[F] = 1.25 * sum_{i<=k} p_i / (k + 0.25 * (sum_i p_i + m))
   where m is the expected number of true matches the candidate set missed; predicting
   nothing scores 1 only if the entity is a singleton, with probability prod_i (1 - p_i) * s.
   A floor on individual probabilities is applied first.
"""
import numpy as np
import polars as pl


def assign(pairs: pl.DataFrame, score: str = "p") -> pl.DataFrame:
    """Keeps each query's best candidate. pairs: (q_rid, s1_rid, score)."""
    return (pairs.sort(["q_rid", score], descending=[False, True])
                 .unique("q_rid", keep="first", maintain_order=True))


def select_threshold(assigned: pl.DataFrame, t: float, score: str = "p") -> pl.DataFrame:
    return assigned.filter(pl.col(score) >= t).select("s1_rid", "q_rid")


def select_expected_f(assigned: pl.DataFrame, floor: float, missing: float, empty_bias: float,
                      score: str = "p") -> pl.DataFrame:
    a = assigned.filter(pl.col(score) >= floor).sort(["s1_rid", score], descending=[False, True])
    if a.height == 0:
        return a.select("s1_rid", "q_rid")
    s1 = a["s1_rid"].to_numpy()
    p = a[score].to_numpy().astype(np.float64)
    # group boundaries
    starts = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]])
    ends = np.r_[starts[1:], len(s1)]
    keep = np.zeros(len(s1), dtype=bool)
    for st, en in zip(starts, ends):
        pp = p[st:en]
        total = pp.sum() + missing
        cum = np.cumsum(pp)
        k = np.arange(1, len(pp) + 1)
        ef = 1.25 * cum / (k + 0.25 * total)
        best = int(np.argmax(ef))
        p_empty = float(np.prod(1.0 - pp)) * empty_bias
        if ef[best] > p_empty:
            keep[st:st + best + 1] = True
    return a.filter(pl.Series(keep)).select("s1_rid", "q_rid")
