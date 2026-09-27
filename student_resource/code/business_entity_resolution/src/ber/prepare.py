"""Load raw TSVs, learn the Indic lexicon from training matches, normalize all records."""
import logging
from multiprocessing import Pool

import polars as pl

from .config import N_JOBS, Paths
from .normalize import normalize_record
from .translit import IndicLexicon

log = logging.getLogger(__name__)

_INDIC_PATTERN = r"[\x{0900}-\x{0D7F}]"


def read_source(paths: Paths, split: str, src: int) -> pl.DataFrame:
    df = pl.read_csv(paths.source_file(split, src), separator="\t", quote_char=None,
                     infer_schema_length=0, missing_utf8_is_empty_string=False)
    return df.with_columns(pl.lit(src, dtype=pl.UInt8).alias("src"))


def read_ground_truth(paths: Paths) -> pl.DataFrame:
    """Long format (s1_id, q_id) of all true matches."""
    gt = pl.read_csv(paths.ground_truth, separator="\t", quote_char=None, infer_schema_length=0)
    return (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
              .explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
              .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "q_id"}))


def learn_lexicon(paths: Paths) -> IndicLexicon:
    s1 = read_source(paths, "train", 1).select(
        pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("n1"),
        pl.col("business_address").alias("a1"))
    q = pl.concat([read_source(paths, "train", s) for s in (2, 3)]).select(
        pl.col("entity_id").alias("q_id"), pl.col("business_name").alias("nq"),
        pl.col("business_address").alias("aq"))
    q = q.filter(pl.col("nq").str.contains(_INDIC_PATTERN)
                 | pl.col("aq").fill_null("").str.contains(_INDIC_PATTERN))
    pairs = read_ground_truth(paths).join(q, on="q_id").join(s1, on="s1_id")
    log.info("learning Indic lexicon from %d matched pairs", pairs.height)
    lex = IndicLexicon.learn(pairs.select("n1", "nq", "a1", "aq").iter_rows())
    log.info("lexicon: %d tokens, %d address components", len(lex.tokens), len(lex.components))
    return lex


_LEX = None


def _init_worker(lex_path):
    global _LEX
    _LEX = IndicLexicon.load(lex_path)


_LIST_COLS = {"name_phones", "addr_nums", "addr_alpha", "addr_phones"}


def _normalize_chunk(df: pl.DataFrame) -> pl.DataFrame:
    cols = None
    for n, a, c in zip(df["business_name"].to_list(), df["business_address"].to_list(),
                       df["country"].to_list()):
        d = normalize_record(n, a, c, _LEX)
        if cols is None:
            cols = {k: [] for k in d}
        for k, v in d.items():
            cols[k].append(v)
    norm = pl.DataFrame(cols, schema_overrides={k: pl.List(pl.Utf8) for k in _LIST_COLS})
    return pl.concat([df, norm], how="horizontal")


def prepare_split(paths: Paths, split: str, lex_path, chunk=50_000):
    """Writes work/<split>/src{1,2,3}.parquet with a split-wide integer id `rid`.

    Chunks stream through the worker pool and straight to disk so the parent never holds
    more than a few chunks in memory.
    """
    offset = 0
    with Pool(N_JOBS, initializer=_init_worker, initargs=(str(lex_path),)) as pool:
        for src in (1, 2, 3):
            df = read_source(paths, split, src)
            df = df.with_columns(
                (pl.int_range(0, pl.len(), dtype=pl.UInt32) + offset).alias("rid"),
                pl.col("country").fill_null("").str.to_lowercase().alias("country"))
            offset += df.height
            log.info("normalizing %s source%d (%d rows)", split, src, df.height)
            tmp = paths.work(split, f"_tmp_src{src}")
            tmp.mkdir(parents=True, exist_ok=True)
            chunks = (df.slice(i, chunk) for i in range(0, df.height, chunk))
            for j, part in enumerate(pool.imap(_normalize_chunk, chunks)):
                part.write_parquet(tmp / f"{j:05d}.parquet")
            del df
            pl.scan_parquet(tmp / "*.parquet").sink_parquet(paths.work(split, f"src{src}.parquet"))
            for f in tmp.glob("*.parquet"):
                f.unlink()
            tmp.rmdir()


def load_records(paths: Paths, split: str, srcs=(1, 2, 3), columns=None, country=None) -> pl.DataFrame:
    frames = []
    for s in srcs:
        lf = pl.scan_parquet(paths.work(split, f"src{s}.parquet"))
        if country is not None:
            lf = lf.filter(pl.col("country") == country)
        if columns is not None:
            lf = lf.select(columns)
        frames.append(lf.collect())
    return pl.concat(frames)


def countries(paths: Paths, split: str):
    return (pl.scan_parquet(paths.work(split, "src1.parquet")).select("country").unique()
              .collect()["country"].sort().to_list())
