"""Official metric: F_0.5 per Source-1 entity, macro-averaged over all Source-1 entities."""
import polars as pl

BETA2 = 0.25


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> dict:
    """pred / truth: long format (s1_id, q_id). s1_ids: every Source-1 id being evaluated."""
    base = pl.DataFrame({"s1_id": s1_ids})
    tp = pred.join(truth, on=["s1_id", "q_id"]).group_by("s1_id").len().rename({"len": "tp"})
    npred = pred.group_by("s1_id").len().rename({"len": "npred"})
    ntrue = truth.group_by("s1_id").len().rename({"len": "ntrue"})
    df = (base.join(tp, on="s1_id", how="left").join(npred, on="s1_id", how="left")
              .join(ntrue, on="s1_id", how="left").fill_null(0))
    f = (pl.when((pl.col("ntrue") == 0) & (pl.col("npred") == 0)).then(1.0)
           .when(pl.col("tp") == 0).then(0.0)
           .otherwise((1 + BETA2) * pl.col("tp") / ((1 + BETA2) * pl.col("tp") + BETA2 * (pl.col("ntrue") - pl.col("tp"))
                                                   + (pl.col("npred") - pl.col("tp")))))
    df = df.with_columns(f.alias("f"))
    tot_tp, tot_pred, tot_true = df["tp"].sum(), df["npred"].sum(), df["ntrue"].sum()
    single = df.filter(pl.col("ntrue") == 0)
    return {
        "macro_f05": df["f"].mean(),
        "micro_precision": tot_tp / max(tot_pred, 1),
        "micro_recall": tot_tp / max(tot_true, 1),
        "singleton_acc": single["f"].mean() if single.height else float("nan"),
        "n_entities": df.height,
    }
