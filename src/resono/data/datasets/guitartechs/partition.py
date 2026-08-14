"""Train/valid/test splits for Guitar-TECHS.

Unlike guitarset, the default here puts **everything in train**. Guitar-TECHS
is intended as extra training material for a model evaluated on GuitarSet, so
holding tracks out of it would shrink the training set to measure nothing —
the evaluation lives in another dataset entirely.

That is a deliberate departure from the player-held-out discipline used for
guitarset, where the same dataset supplies both training and evaluation and a
leak would flatter the numbers. Pass ``held_out_player`` when Guitar-TECHS is
the thing being measured rather than the thing being learned from.
"""
import json
from pathlib import Path

DATASET_NAME = "guitartechs"

# Player 03 contributed only the 12 musical excerpts; players 01 and 02
# recorded everything else. So holding out P3 removes the only realistic
# playing in the set, and holding out P1 or P2 removes half of everything.
# Neither is a clean split, which is a further reason to evaluate elsewhere.
PLAYERS = ("P1", "P2", "P3")


def partition(
    cache_dir: Path,
    partitions_dir: Path,
    held_out_player: str | None = None,
) -> None:
    """Write guitartechs.json listing which stems go to which split.

    Parameters
    ----------
    held_out_player:
        One of 'P1', 'P2', 'P3'. That player's takes become valid and test
        (the same list, since there is not enough material to split further);
        everything else trains. Default None puts every stem in train.
    """
    root = Path(cache_dir) / DATASET_NAME
    stems = sorted(p.name[:-len("-audio.npy")] for p in root.glob("*-audio.npy"))
    if not stems:
        raise FileNotFoundError(f"No preprocessed files found in {root}")

    if held_out_player is None:
        result = {"train": stems, "valid": [], "test": []}
    else:
        if held_out_player not in PLAYERS:
            raise ValueError(
                f"held_out_player must be one of {PLAYERS}, got {held_out_player!r}"
            )
        held = [s for s in stems if s.startswith(f"{held_out_player}_")]
        rest = [s for s in stems if not s.startswith(f"{held_out_player}_")]
        if not held:
            raise ValueError(f"No cached stems for player {held_out_player}")
        result = {"train": rest, "valid": held, "test": held}

    Path(partitions_dir).mkdir(parents=True, exist_ok=True)
    out = Path(partitions_dir) / f"{DATASET_NAME}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"Partition saved → {out}: {len(result['train'])} train / "
        f"{len(result['valid'])} valid / {len(result['test'])} test"
        + ("" if held_out_player is None else f"  (held out {held_out_player})")
    )
