"""Candidate generation.

Every record emits hashed blocking keys of several types (see KEY_TYPES). A query (S2/S3
record) is paired with every Source-1 record of the same country that shares at least one
sufficiently specific key; candidates are ranked by the IDF-weighted sum of shared keys and
the top-K per query are kept. Blocking runs query-side because each S2/S3 record matches at
most one Source-1 entity.
"""
import math

import polars as pl

from .config import BlockingParams

# key family -> bit id (single-token families are capped harder, see SINGLE_TYPES)
KEY_TYPES = {
    "T": 0,   # name core token
    "S": 1,   # name token consonant skeleton
    "P": 2,   # pair of name core tokens
    "C": 3,   # concatenated name (catches domains / spacing noise)
    "A": 4,   # address number + address word
    "N": 5,   # name token + address word
    "Q": 6,   # phone number
    "W": 7,   # pair of address words
    "R": 8,   # triple of name core tokens
    "K": 9,   # pair of name skeletons
    "M": 10,  # name token + address number
    "X": 11,  # prefix / suffix of concatenated name
    "U": 12,  # pair of address numbers
}
SINGLE_TYPES = {0, 1}
KEY_COLS = ["rid", "src", "country", "name_core", "name_skel", "name_cat", "name_full_cat",
            "addr_nums", "addr_alpha", "name_phones", "addr_phones"]


def _tok_list(col, max_n):
    return (pl.col(col).str.split(" ")
            .list.eval(pl.element().filter((pl.element().str.len_chars() >= 2)
                                           & ~pl.element().str.contains(r"^\d+$")))
            .list.unique(maintain_order=True).list.head(max_n))


def _explode_pos(df, col, alias):
    return (df.select("rid", "country", pl.col(col).alias(alias))
              .with_columns(pl.int_ranges(0, pl.col(alias).list.len()).alias(alias + "_pos"))
              .explode([alias, alias + "_pos"]).drop_nulls(alias))


def record_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Returns (rid, kt, key) with key a u64 hash scoped by country."""
    base = df.select(
        "rid", "country",
        _tok_list("name_core", 6).alias("nt"),
        pl.col("name_skel").str.split(" ")
          .list.eval(pl.element().filter(pl.element().str.len_chars() >= 3))
          .list.unique().list.head(6).alias("sk"),
        pl.col("name_skel").str.split(" ")
          .list.eval(pl.element().filter(pl.element().str.len_chars() >= 1))
          .list.unique(maintain_order=True).list.head(5).alias("sk2"),
        pl.col("name_cat"), pl.col("name_full_cat"),
        pl.col("addr_nums").list.eval(pl.element().filter(pl.element().str.len_chars() <= 6))
          .list.head(3).alias("an"),
        pl.col("addr_alpha").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3))
          .list.head(5).alias("aa"),
        pl.concat_list(pl.col("name_phones"), pl.col("addr_phones")).list.unique().alias("ph"),
    )
    parts = []

    def emit(frame, kt, expr):
        parts.append(frame.select(
            "rid", pl.lit(kt, dtype=pl.UInt8).alias("kt"),
            pl.concat_str([pl.col("country"), pl.lit(f"|{kt}|"), expr]).hash(seed=7).alias("key")))

    nt = _explode_pos(base, "nt", "t")
    emit(nt, KEY_TYPES["T"], pl.col("t"))
    sk = base.select("rid", "country", pl.col("sk")).explode("sk").drop_nulls("sk")
    emit(sk, KEY_TYPES["S"], pl.col("sk"))
    pairs = nt.join(nt.select("rid", pl.col("t").alias("t2"), pl.col("t_pos").alias("p2")), on="rid")
    pairs = pairs.filter(pl.col("t") < pl.col("t2"))
    emit(pairs, KEY_TYPES["P"], pl.concat_str([pl.col("t"), pl.col("t2")], separator="|"))
    n4 = nt.filter(pl.col("t_pos") < 4)
    trip = (n4.join(n4.select("rid", pl.col("t").alias("t2")), on="rid").filter(pl.col("t") < pl.col("t2"))
              .join(n4.select("rid", pl.col("t").alias("t3")), on="rid").filter(pl.col("t2") < pl.col("t3")))
    emit(trip, KEY_TYPES["R"], pl.concat_str([pl.col("t"), pl.col("t2"), pl.col("t3")], separator="|"))
    sk2 = base.select("rid", "country", pl.col("sk2")).explode("sk2").drop_nulls("sk2")
    sk2 = sk2.join(sk2.select("rid", pl.col("sk2").alias("s2")), on="rid").filter(pl.col("sk2") < pl.col("s2"))
    emit(sk2, KEY_TYPES["K"], pl.concat_str([pl.col("sk2"), pl.col("s2")], separator="|"))
    cat = pl.concat([
        base.select("rid", "country", pl.col("name_cat").alias("c")),
        base.select("rid", "country", pl.col("name_full_cat").alias("c")),
    ]).filter(pl.col("c").str.len_chars() >= 4).unique(["rid", "c"])
    emit(cat, KEY_TYPES["C"], pl.col("c"))
    long_cat = cat.filter(pl.col("c").str.len_chars() >= 12)
    affix = pl.concat([
        long_cat.select("rid", "country", pl.concat_str([pl.lit("<"), pl.col("c").str.head(10)]).alias("c")),
        long_cat.select("rid", "country", pl.concat_str([pl.lit(">"), pl.col("c").str.tail(10)]).alias("c")),
    ]).unique(["rid", "c"])
    emit(affix, KEY_TYPES["X"], pl.col("c"))
    an = base.select("rid", "country", pl.col("an")).explode("an").drop_nulls("an")
    emit(nt.filter(pl.col("t_pos") < 3).join(an.select("rid", "an"), on="rid"), KEY_TYPES["M"],
         pl.concat_str([pl.col("t"), pl.col("an")], separator="|"))
    nn = an.join(an.select("rid", pl.col("an").alias("an2")), on="rid").filter(pl.col("an") < pl.col("an2"))
    emit(nn, KEY_TYPES["U"], pl.concat_str([pl.col("an"), pl.col("an2")], separator="|"))
    aa = _explode_pos(base, "aa", "a")
    emit(an.join(aa.select("rid", "a"), on="rid"), KEY_TYPES["A"],
         pl.concat_str([pl.col("an"), pl.col("a")], separator="|"))
    emit(nt.filter(pl.col("t_pos") < 3).join(aa.select("rid", "a"), on="rid"), KEY_TYPES["N"],
         pl.concat_str([pl.col("t"), pl.col("a")], separator="|"))
    ww = aa.filter(pl.col("a_pos") < 4)
    ww = ww.join(ww.select("rid", pl.col("a").alias("a2")), on="rid").filter(pl.col("a") < pl.col("a2"))
    emit(ww, KEY_TYPES["W"], pl.concat_str([pl.col("a"), pl.col("a2")], separator="|"))
    ph = base.select("rid", "country", pl.col("ph")).explode("ph").drop_nulls("ph")
    emit(ph, KEY_TYPES["Q"], pl.col("ph"))
    return pl.concat(parts).unique(["rid", "key"])


def record_keys_chunked(df: pl.DataFrame, chunk: int = 150_000) -> pl.DataFrame:
    return pl.concat([record_keys(df.slice(i, chunk)) for i in range(0, df.height, chunk)])


def build_index(s1: pl.DataFrame, params: BlockingParams) -> pl.DataFrame:
    """Source-1 postings (key, kt, s1_rid, w) restricted to specific-enough keys."""
    keys = record_keys_chunked(s1)
    df = keys.group_by("key").agg(pl.len().alias("df"), pl.col("kt").first())
    n = max(s1.height, 1)
    cap = (pl.when(pl.col("kt").is_in(list(SINGLE_TYPES)))
             .then(pl.lit(params.cap_single)).otherwise(pl.lit(params.cap_pair)))
    df = df.filter(pl.col("df") <= cap).with_columns(
        (pl.lit(math.log(n)) - pl.col("df").cast(pl.Float32).log()).cast(pl.Float32).alias("w"))
    return keys.join(df.select("key", "w"), on="key").rename({"rid": "s1_rid"})


def query_candidates(q: pl.DataFrame, index: pl.DataFrame, params: BlockingParams) -> pl.DataFrame:
    qk = record_keys_chunked(q).rename({"rid": "q_rid"})
    hits = qk.join(index.select("key", "s1_rid", "w"), on="key").with_columns(
        pl.col("kt").replace_strict({v: 1 << v for v in KEY_TYPES.values()},
                                    return_dtype=pl.UInt16).alias("bit"))
    cand = (hits.group_by("q_rid", "s1_rid")
                .agg(pl.col("w").sum().alias("bscore"),
                     pl.col("bit").unique().sum().cast(pl.UInt16).alias("ktmask"),
                     pl.len().cast(pl.UInt16).alias("nkeys")))
    cand = cand.with_columns(
        pl.col("bscore").rank("ordinal", descending=True).over("q_rid").cast(pl.UInt16).alias("brank"),
        pl.col("bscore").max().over("q_rid").alias("bmax"))
    return cand.filter((pl.col("brank") <= params.max_candidates)
                       & (pl.col("bscore") >= params.min_rel_score * pl.col("bmax")))

