import json
from pathlib import Path

import numpy as np


def partition(
    cache_dir: Path,
    partitions_dir: Path,
    split_by_player: bool = True,
    val_players: list[str] | None = None,
    test_players: list[str] | None = None,
    seed: int = 42,
    name: str = "guitarset",
) -> None:
    """Write train/valid/test split JSON for GuitarSet.

    Player-based split (default): 4 players → train, 1 → valid, 1 → test.
    Random split: 70 / 15 / 15 %.

    The stem for each track is its track ID (e.g. '00_BN1-129-Eb_solo'),
    derived by stripping the '_mic' suffix from the audio filename.

    Parameters
    ----------
    name:
        Dataset name, used both for the cache subdirectory and the output JSON
        filename. Exists so that GuitarSet-derived datasets — relabelled
        variants that share these stems and this player numbering — can reuse
        this split rather than copy it: two datasets are only comparable if
        they are split identically, which a duplicate implementation cannot
        guarantee over time.
    """
    # GuitarSet has 6 players (IDs '00'–'05'), each contributing 60 tracks
    # (6 chord types × 5 styles × 2 tempos). Assigning the last two players
    # to validation and test gives a clean 240/60/60 split with no player
    # identity leaking across splits — a requirement for fair generalisation.
    # Players '04' and '05' are chosen by convention (same as FretNet / polypenn).
    if val_players is None:
        val_players = ["04"]
    if test_players is None:
        test_players = ["05"]

    gset_cache = Path(cache_dir) / name
    stems = sorted(
        p.stem[:-6]                          # strip '-audio' (6 chars)
        for p in gset_cache.glob("*-audio.npy")
    )
    if not stems:
        raise FileNotFoundError(f"No preprocessed files found in {gset_cache}")

    if split_by_player:
        train, valid, test = [], [], []
        for stem in stems:
            player = stem[:2]
            if player in test_players:
                test.append(stem)
            elif player in val_players:
                valid.append(stem)
            else:
                train.append(stem)
    else:
        rng = np.random.default_rng(seed)
        shuffled = [stems[i] for i in rng.permutation(len(stems))]
        n      = len(shuffled)
        n_test = max(1, int(n * 0.15))
        n_val  = max(1, int(n * 0.15))
        test   = shuffled[:n_test]
        valid  = shuffled[n_test : n_test + n_val]
        train  = shuffled[n_test + n_val :]

    result = {
        "train": sorted(train),
        "valid": sorted(valid),
        "test":  sorted(test),
    }

    Path(partitions_dir).mkdir(parents=True, exist_ok=True)
    out = Path(partitions_dir) / f"{name}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"Partition saved → {out}: "
        f"{len(train)} train / {len(valid)} valid / {len(test)} test"
    )


def cv_folds(
    cache_dir: Path,
    partitions_dir: Path,
    name: str = "guitarset",
) -> None:
    """Write one partition JSON per 6-fold cross-validation rotation.

    Each fold assigns one player to test, the next player (cyclically) to
    valid, and the remaining four to train — matching FretNet's evaluation
    protocol. This removes the variance introduced by any single fixed
    choice of val/test players and lets you average metrics across all folds.

    Writes {name}_fold0.json … {name}_fold5.json to partitions_dir. These are
    independent of the default {name}.json produced by partition(). See
    :func:`partition` for what ``name`` is for.
    """
    all_players = ["00", "01", "02", "03", "04", "05"]

    gset_cache = Path(cache_dir) / name
    stems = sorted(
        p.stem[:-6]
        for p in gset_cache.glob("*-audio.npy")
    )
    if not stems:
        raise FileNotFoundError(f"No preprocessed files found in {gset_cache}")

    by_player: dict[str, list[str]] = {p: [] for p in all_players}
    for stem in stems:
        by_player[stem[:2]].append(stem)

    Path(partitions_dir).mkdir(parents=True, exist_ok=True)

    for fold, test_player in enumerate(all_players):
        # Validation is the next player in the rotation so the two held-out
        # players are always adjacent — the same pairing strategy as the fixed
        # default split, just rotated systematically through all 6 positions.
        val_player = all_players[(fold + 1) % len(all_players)]

        test  = sorted(by_player[test_player])
        valid = sorted(by_player[val_player])
        train = sorted(
            stem for p, tracks in by_player.items()
            if p not in (test_player, val_player)
            for stem in tracks
        )

        result = {"train": train, "valid": valid, "test": test}
        out = Path(partitions_dir) / f"{name}_fold{fold}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2)

        print(
            f"Fold {fold} saved → {out}: "
            f"test={test_player}, valid={val_player}, "
            f"{len(train)} train / {len(valid)} valid / {len(test)} test"
        )
