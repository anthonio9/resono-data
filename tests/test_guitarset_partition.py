import json

import numpy as np
import pytest

from resono.data.datasets.guitarset.partition import cv_folds, partition


def _make_cache(tmp_path, stems):
    cache = tmp_path / "cache" / "guitarset"
    cache.mkdir(parents=True)
    for stem in stems:
        np.save(cache / f"{stem}-audio.npy", np.zeros(256, dtype=np.float32))
    return tmp_path / "cache"


def _read(partitions_dir):
    with open(partitions_dir / "guitarset.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# partition() — player split
# ---------------------------------------------------------------------------

def test_player_split_counts(tmp_path):
    # 6 players × 10 tracks each = 60 tracks total.
    stems = [f"{p:02d}_track_{i:02d}" for p in range(6) for i in range(10)]
    cache_dir      = _make_cache(tmp_path, stems)
    partitions_dir = tmp_path / "partitions"
    partitions_dir.mkdir()

    partition(cache_dir, partitions_dir, split_by_player=True,
              val_players=["04"], test_players=["05"])
    p = _read(partitions_dir)

    assert len(p["train"]) == 40   # players 00–03
    assert len(p["valid"]) == 10   # player 04
    assert len(p["test"])  == 10   # player 05


def test_player_split_no_leakage(tmp_path):
    stems = [f"{p:02d}_track_{i:02d}" for p in range(6) for i in range(10)]
    cache_dir      = _make_cache(tmp_path, stems)
    partitions_dir = tmp_path / "partitions"
    partitions_dir.mkdir()

    partition(cache_dir, partitions_dir, split_by_player=True,
              val_players=["04"], test_players=["05"])
    p = _read(partitions_dir)

    for stem in p["train"]:
        assert not stem.startswith("04") and not stem.startswith("05")
    for stem in p["valid"]:
        assert stem.startswith("04")
    for stem in p["test"]:
        assert stem.startswith("05")


# ---------------------------------------------------------------------------
# partition() — random split
# ---------------------------------------------------------------------------

def test_random_split_total(tmp_path):
    stems = [f"track_{i:03d}" for i in range(100)]
    cache_dir      = _make_cache(tmp_path, stems)
    partitions_dir = tmp_path / "partitions"
    partitions_dir.mkdir()

    partition(cache_dir, partitions_dir, split_by_player=False, seed=42)
    p = _read(partitions_dir)

    total = len(p["train"]) + len(p["valid"]) + len(p["test"])
    assert total == 100


def test_random_split_ratios(tmp_path):
    stems = [f"track_{i:03d}" for i in range(100)]
    cache_dir      = _make_cache(tmp_path, stems)
    partitions_dir = tmp_path / "partitions"
    partitions_dir.mkdir()

    partition(cache_dir, partitions_dir, split_by_player=False, seed=42)
    p = _read(partitions_dir)

    assert 12 <= len(p["test"])  <= 18   # ~15 %
    assert 12 <= len(p["valid"]) <= 18


def test_random_split_reproducible(tmp_path):
    stems = [f"track_{i:03d}" for i in range(100)]
    cache_dir = _make_cache(tmp_path, stems)

    pd1 = tmp_path / "p1"
    pd2 = tmp_path / "p2"
    pd1.mkdir(); pd2.mkdir()

    partition(cache_dir, pd1, split_by_player=False, seed=7)
    partition(cache_dir, pd2, split_by_player=False, seed=7)

    with open(pd1 / "guitarset.json") as f: p1 = json.load(f)
    with open(pd2 / "guitarset.json") as f: p2 = json.load(f)
    assert p1 == p2


# ---------------------------------------------------------------------------
# cv_folds()
# ---------------------------------------------------------------------------

def test_cv_folds_creates_six_files(tmp_path):
    stems = [f"{p:02d}_track_{i:02d}" for p in range(6) for i in range(10)]
    cache_dir      = _make_cache(tmp_path, stems)
    partitions_dir = tmp_path / "partitions"
    partitions_dir.mkdir()

    cv_folds(cache_dir, partitions_dir)

    for fold in range(6):
        assert (partitions_dir / f"guitarset_fold{fold}.json").exists()


def test_cv_folds_no_leakage(tmp_path):
    stems = [f"{p:02d}_track_{i:02d}" for p in range(6) for i in range(10)]
    cache_dir      = _make_cache(tmp_path, stems)
    partitions_dir = tmp_path / "partitions"
    partitions_dir.mkdir()

    cv_folds(cache_dir, partitions_dir)

    for fold in range(6):
        with open(partitions_dir / f"guitarset_fold{fold}.json") as f:
            p = json.load(f)
        train_stems = set(p["train"])
        valid_stems = set(p["valid"])
        test_stems  = set(p["test"])

        assert not (train_stems & valid_stems), "Train/valid overlap in fold"
        assert not (train_stems & test_stems),  "Train/test overlap in fold"
        assert not (valid_stems & test_stems),  "Valid/test overlap in fold"


def test_cv_folds_all_players_appear_as_test(tmp_path):
    stems = [f"{p:02d}_track_{i:02d}" for p in range(6) for i in range(10)]
    cache_dir      = _make_cache(tmp_path, stems)
    partitions_dir = tmp_path / "partitions"
    partitions_dir.mkdir()

    cv_folds(cache_dir, partitions_dir)

    test_players = set()
    for fold in range(6):
        with open(partitions_dir / f"guitarset_fold{fold}.json") as f:
            p = json.load(f)
        for stem in p["test"]:
            test_players.add(stem[:2])

    assert test_players == {"00", "01", "02", "03", "04", "05"}


# ---------------------------------------------------------------------------
# name= — sharing this split with GuitarSet-derived datasets
# ---------------------------------------------------------------------------

def test_name_selects_cache_dir_and_output_file(tmp_path):
    """A derived dataset splits through the same code, under its own name.

    reguitarset holds the same tracks under the same stems, so it must be split
    identically for any comparison against guitarset to mean anything. That is
    what this parameter is for; a second implementation would drift.
    """
    stems = [f"{p:02d}_track_{i:02d}" for p in range(6) for i in range(10)]

    cache_dir = tmp_path / "cache"
    (cache_dir / "reguitarset").mkdir(parents=True)
    for stem in stems:
        np.save(cache_dir / "reguitarset" / f"{stem}-audio.npy",
                np.zeros(256, dtype=np.float32))

    partitions_dir = tmp_path / "partitions"
    partitions_dir.mkdir()

    partition(cache_dir, partitions_dir, name="reguitarset")
    cv_folds(cache_dir, partitions_dir, name="reguitarset")

    assert (partitions_dir / "reguitarset.json").exists()
    assert not (partitions_dir / "guitarset.json").exists()
    for fold in range(6):
        assert (partitions_dir / f"reguitarset_fold{fold}.json").exists()


def test_name_default_matches_an_explicit_guitarset(tmp_path):
    """The default is inert: omitting name= is the same as passing 'guitarset'."""
    stems = [f"{p:02d}_track_{i:02d}" for p in range(6) for i in range(10)]
    cache_dir = _make_cache(tmp_path, stems)

    default = tmp_path / "default"
    explicit = tmp_path / "explicit"
    default.mkdir()
    explicit.mkdir()

    partition(cache_dir, default)
    partition(cache_dir, explicit, name="guitarset")

    with open(default / "guitarset.json") as f:
        assert json.load(f) == json.load(open(explicit / "guitarset.json"))
