"""Train/valid/test split for GOAT.

Every cached take goes to train; valid and test are left empty. That is the
same choice the guitartechs module makes, for the same reason: GOAT's shipped
test split is 7 of 172 takes, which is far too small to measure on, and the
corpus is 77% one player and 63% one guitar, so any split drawn from it would
measure that player rather than the model. GuitarSet stays the yardstick.

Pass ``held_out_player`` if GOAT itself has to be evaluated on — it moves one
player's takes wholesale into valid and test (the same list, there being too
little material to divide further).
"""
import json
from pathlib import Path

from resono.data.datasets.goat.preprocess import DATASET_NAME


def partition(
    cache_dir: Path,
    partitions_dir: Path,
    name: str = DATASET_NAME,
    held_out_player: str | None = None,
    raw_dir: Path = Path("data/raw"),
) -> None:
    """Write {name}.json listing which takes go to which split."""
    cache = Path(cache_dir) / name
    stems = sorted(
        path.name[: -len("-audio.npy")] for path in cache.glob("*-audio.npy")
    )
    if not stems:
        raise FileNotFoundError(f"no cached takes under {cache}")

    if held_out_player is None:
        result = {"train": stems, "valid": [], "test": []}
    else:
        players = _players(Path(raw_dir))
        held = [stem for stem in stems if players.get(stem) == held_out_player]
        rest = [stem for stem in stems if players.get(stem) != held_out_player]
        if not held:
            raise ValueError(f"no cached takes for player {held_out_player!r}")
        result = {"train": rest, "valid": held, "test": held}

    Path(partitions_dir).mkdir(parents=True, exist_ok=True)
    out = Path(partitions_dir) / f"{name}.json"
    out.write_text(json.dumps(result, indent=2))
    print(
        f"{out}: {len(result['train'])} train / {len(result['valid'])} valid / "
        f"{len(result['test'])} test"
        + ("" if held_out_player is None else f"  (held out {held_out_player})")
    )


def _players(raw_dir: Path) -> dict[str, str]:
    import csv
    from resono.data.datasets.goat.preprocess import GOAT_DIRNAME
    path = raw_dir / GOAT_DIRNAME / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is needed to split by player")
    return {row["item"]: row["player"] for row in csv.DictReader(path.open())}
