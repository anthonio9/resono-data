"""Build the pickup cache, reusing guitarset's label pipeline unchanged."""
from pathlib import Path

from resono.data.datasets.guitarset.preprocess import preprocess as preprocess_guitarset

DATASET_NAME = "guitarset-pickup"


def preprocess(raw_dir: Path, cache_dir: Path, **kwargs) -> None:
    """Convert the pickup recordings to .npy cache.

    Accepts everything :func:`guitarset.preprocess.preprocess` does — the same
    grid, the same tail treatment — with the audio source and output name
    fixed, so the two caches cannot collide.
    """
    kwargs.pop("audio_source", None)
    kwargs.pop("dataset_name", None)
    preprocess_guitarset(
        raw_dir, cache_dir,
        audio_source="pickup", dataset_name=DATASET_NAME, **kwargs
    )
