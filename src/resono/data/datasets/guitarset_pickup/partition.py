"""Split the pickup cache exactly as the microphone cache is split.

The two datasets are the *same performances* recorded through two transducers.
If they were split independently, a performance could appear as microphone in
train and as pickup in test — different audio, identical notes, identical
labels, and a model that had already seen the answer. That is a real leak and
an easy one to introduce by accident, so this module does not compute a split
at all where one already exists: it mirrors guitarset's.

Mirroring is safe because the stems are the same on both sides. GuitarSet's
default split is player-held-out (00-03 train, 04 valid, 05 test) and derived
deterministically from the stem prefix, so recomputing it would give the same
answer — but only while the defaults are unchanged. Copying holds even if the
microphone split was made with different players.
"""
import json
from pathlib import Path

from resono.data.datasets.guitarset.partition import partition as partition_guitarset

DATASET_NAME = "guitarset-pickup"
SOURCE_NAME = "guitarset"


def partition(
    cache_dir: Path,
    partitions_dir: Path,
    mirror: bool = True,
    **kwargs,
) -> None:
    """Write guitarset-pickup.json.

    Parameters
    ----------
    mirror:
        Copy guitarset.json verbatim when it exists, so both audio sources of
        a performance land in the same split. Turning this off computes an
        independent split, which is only safe if the microphone variant is not
        also in use.
    """
    source = Path(partitions_dir) / f"{SOURCE_NAME}.json"

    if not (mirror and source.exists()):
        if mirror:
            print(
                f"{source} does not exist yet, so there is nothing to mirror — "
                "computing the split directly. Both datasets will still agree, "
                "because the player split is deterministic."
            )
        partition_guitarset(
            cache_dir, partitions_dir, name=DATASET_NAME, **kwargs
        )
        return

    with open(source) as f:
        split = json.load(f)

    cached = {
        p.name[: -len("-audio.npy")]
        for p in (Path(cache_dir) / DATASET_NAME).glob("*-audio.npy")
    }
    if not cached:
        raise FileNotFoundError(
            f"No preprocessed files found in {Path(cache_dir) / DATASET_NAME}"
        )

    # A mirrored split names microphone stems; the pickup cache must contain
    # them all, or training would fail on the first missing one.
    listed = {stem for stems in split.values() for stem in stems}
    missing = listed - cached
    if missing:
        raise ValueError(
            f"{len(missing)} stems in {source.name} are not in the pickup cache "
            f"(e.g. {sorted(missing)[:3]}). Build it for the same tracks first, "
            "or pass mirror=False to split what is actually there."
        )

    out = Path(partitions_dir) / f"{DATASET_NAME}.json"
    Path(partitions_dir).mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(split, f, indent=2)

    extra = len(cached - listed)
    print(
        f"Partition mirrored from {source.name} → {out}: "
        f"{len(split['train'])} train / {len(split['valid'])} valid / "
        f"{len(split['test'])} test"
        + (f"  ({extra} cached stems not in the split)" if extra else "")
    )
