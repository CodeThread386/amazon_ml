"""LightGBM helpers: training, cross-fitting and streamed prediction."""
import logging

import lightgbm as lgb
import numpy as np
import polars as pl

log = logging.getLogger(__name__)

ID_COLS = ("q_rid", "s1_rid", "label", "fold")


def feature_names(df: pl.DataFrame):
    return [c for c in df.columns if c not in ID_COLS]


def fold_of(s1_rid: pl.Expr, n_folds: int) -> pl.Expr:
    return (s1_rid.hash(seed=3) % n_folds).cast(pl.UInt8)


def train_lgb(df: pl.DataFrame, params: dict, rounds: int, valid: pl.DataFrame = None,
              features=None) -> lgb.Booster:
    features = features or feature_names(df)
    x = df.select(features).to_numpy().astype(np.float32, copy=False)
    y = df["label"].to_numpy()
    dtrain = lgb.Dataset(x, y, feature_name=list(features), free_raw_data=True)
    del x
    sets, names = [dtrain], ["train"]
    if valid is not None:
        dvalid = lgb.Dataset(valid.select(features).to_numpy().astype(np.float32, copy=False),
                             valid["label"].to_numpy(), reference=dtrain)
        sets.append(dvalid)
        names.append("valid")
    p = {**params, "metric": ["binary_logloss", "auc"], "num_threads": params.get("num_threads", 0)}
    callbacks = [lgb.log_evaluation(100)]
    if valid is not None:
        callbacks.append(lgb.early_stopping(50, verbose=False))
    booster = lgb.train(p, dtrain, num_boost_round=rounds, valid_sets=sets, valid_names=names,
                        callbacks=callbacks)
    log.info("trained %d trees on %d rows x %d features", booster.num_trees(), df.height, len(features))
    return booster


def predict(booster: lgb.Booster, df: pl.DataFrame) -> np.ndarray:
    feats = booster.feature_name()
    return booster.predict(df.select(feats).to_numpy().astype(np.float32, copy=False),
                           num_threads=0).astype(np.float32)


def importance(booster: lgb.Booster, top: int = 25):
    gain = booster.feature_importance("gain")
    order = np.argsort(-gain)[:top]
    total = gain.sum() or 1.0
    return [(booster.feature_name()[i], round(float(gain[i] / total), 4)) for i in order]
