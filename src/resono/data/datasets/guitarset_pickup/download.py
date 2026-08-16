"""Fetch the annotations and the mono pickup mix."""
from pathlib import Path

from resono.data.datasets.guitarset.download import download as download_guitarset

# The pickup mix, plus the annotations it shares with the microphone set. The
# raw tree is shared with guitarset — same Zenodo record, same performances —
# so nothing is downloaded twice.
ARCHIVES = ("annotation", "audio_mono-pickup_mix")


def download(raw_dir: Path, rename_sharp: bool = True, progress: bool = True) -> None:
    """Download GuitarSet's pickup audio into raw_dir/guitarset/."""
    download_guitarset(
        raw_dir, archives=ARCHIVES, rename_sharp=rename_sharp, progress=progress
    )
