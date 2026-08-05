"""Train/valid/test splits, shared with GuitarSet by construction.

reguitarset holds the same 360 tracks as guitarset under the same stems, so it
is split by the same code with the dataset name swapped. Sharing rather than
copying is the point: comparing a model trained on the relabelled tails against
one trained on the originals is only meaningful if both saw identical splits,
and a duplicated implementation would drift out of agreement silently.
"""
from pathlib import Path

from resono.data.datasets.guitarset.partition import cv_folds as _cv_folds
from resono.data.datasets.guitarset.partition import partition as _partition

DATASET_NAME = "reguitarset"


def partition(
    cache_dir: Path,
    partitions_dir: Path,
    split_by_player: bool = True,
    val_players: list[str] | None = None,
    test_players: list[str] | None = None,
    seed: int = 42,
) -> None:
    """Write reguitarset.json — see guitarset.partition.partition."""
    _partition(
        cache_dir, partitions_dir,
        split_by_player=split_by_player,
        val_players=val_players,
        test_players=test_players,
        seed=seed,
        name=DATASET_NAME,
    )


def cv_folds(cache_dir: Path, partitions_dir: Path) -> None:
    """Write reguitarset_fold0.json … _fold5.json — see guitarset.partition.cv_folds."""
    _cv_folds(cache_dir, partitions_dir, name=DATASET_NAME)
