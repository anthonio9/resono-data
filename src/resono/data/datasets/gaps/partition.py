import json
from pathlib import Path

import numpy as np

from resono.data.datasets.gaps.download import GAPS_DIRNAME, read_metadata


def partition(
    cache_dir: Path,
    partitions_dir: Path,
    raw_dir: Path = Path("data/raw"),
    valid_fraction: float = 0.1,
    seed: int = 42,
) -> None:
    """Write train/valid/test split JSON for GAPS.

    The authors' own split is kept for test, and validation is carved out of
    their train set by performer so that no performer appears in both.

    A caveat worth knowing: the published split is a random 90:10 division by
    piece, not by performer, and 12 performers appear in both its train and
    test halves. Splitting the validation set by performer removes that leak
    between train and valid but cannot remove it between train and test —
    doing so would mean abandoning the authors' test set, at the cost of
    comparability with published GAPS numbers. Test results are therefore
    slightly optimistic; GuitarSet, whose split is player-disjoint by
    construction, is the stricter generalisation check.

    Parameters
    ----------
    raw_dir:
        Location of the downloaded metadata CSV, which supplies the official
        split and the performer names.
    valid_fraction:
        Approximate share of the official train set to hold out for
        validation. Whole performers move at a time, so the realised fraction
        lands near rather than on this value.
    """
    gaps_cache = Path(cache_dir) / GAPS_DIRNAME
    cached = {
        path.stem[:-6]                        # strip '-audio'
        for path in gaps_cache.glob("*-audio.npy")
    }
    if not cached:
        raise FileNotFoundError(f"No preprocessed files found in {gaps_cache}")

    rows = read_metadata(Path(raw_dir) / GAPS_DIRNAME)
    official = {row["id"]: row["split"].strip() for row in rows}
    performers = {row["id"]: (row.get("performer_name") or "").strip() for row in rows}

    test = sorted(t for t in cached if official.get(t) == "test")
    train_pool = sorted(t for t in cached if official.get(t) == "train")
    if not train_pool:
        raise ValueError("no cached tracks belong to the official train split")

    # Group by performer, keeping unnamed performers as singleton groups so an
    # empty name cannot merge unrelated tracks into one giant block.
    groups: dict[str, list[str]] = {}
    for track in train_pool:
        name = performers.get(track) or f"__unknown__{track}"
        groups.setdefault(name, []).append(track)

    names = sorted(groups)
    rng = np.random.default_rng(seed)
    target = len(train_pool) * valid_fraction

    valid: list[str] = []
    for index in rng.permutation(len(names)):
        if len(valid) >= target:
            break
        valid.extend(groups[names[index]])

    valid_set = set(valid)
    train = [track for track in train_pool if track not in valid_set]

    result = {
        "train": sorted(train),
        "valid": sorted(valid),
        "test": sorted(test),
    }

    Path(partitions_dir).mkdir(parents=True, exist_ok=True)
    out = Path(partitions_dir) / "gaps.json"
    with open(out, "w") as handle:
        json.dump(result, handle, indent=2)

    print(
        f"Partition saved → {out}: "
        f"{len(train)} train / {len(valid)} valid / {len(test)} test"
    )
