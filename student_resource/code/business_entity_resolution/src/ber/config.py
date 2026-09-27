from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class Paths:
    data_dir: Path
    work_dir: Path
    output_dir: Path

    def split_dir(self, split: str) -> Path:
        return self.data_dir / split

    def source_file(self, split: str, src: int) -> Path:
        return self.data_dir / split / f"{split}_source{src}.tsv"

    @property
    def ground_truth(self) -> Path:
        return self.data_dir / "train" / "train_ground_truth.tsv"

    def work(self, *parts: str) -> Path:
        p = self.work_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


@dataclass
class BlockingParams:
    # Keys whose Source-1 document frequency exceeds the cap are too unspecific to block on.
    cap_single: int = 60
    cap_pair: int = 200
    # blocking candidates per query handed to the re-ranker
    max_candidates: int = 30
    # a candidate is kept only if its blocking score is at least this fraction of the query's best
    min_rel_score: float = 0.0
    query_chunk: int = 50_000
    # re-ranker pruning: candidates per query fed to the matching model, and probability floor
    keep_k: int = 6
    rerank_floor: float = 0.002
    rerank_sample_per_country: int = 150_000


@dataclass
class ModelParams:
    n_folds: int = 2
    train_pairs_per_fold: int = 1_500_000
    use_context_model: bool = False
    seed: int = 13
    lgb: dict = field(default_factory=lambda: {
        "objective": "binary",
        "learning_rate": 0.1,
        "num_leaves": 255,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 127,
        "seed": 13,
        "verbose": -1,
    })
    num_boost_round: int = 300


N_JOBS = max(1, (os.cpu_count() or 2) - 1)
