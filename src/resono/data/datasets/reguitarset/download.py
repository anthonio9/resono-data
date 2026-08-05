"""Fetch GuitarSet plus the hexaphonic audio the relabelling needs.

The raw files are shared with the ``guitarset`` dataset — same Zenodo record,
same annotations, same mic audio — so everything lands under
``raw_dir/guitarset/`` and nothing is downloaded twice. Only the caches the two
datasets produce are separate.
"""
from pathlib import Path

from resono.data.datasets.guitarset.download import (
    _ZENODO,
    _download_file,
    extract_and_rename_sharp,
)
from resono.data.datasets.guitarset.download import download as download_guitarset

# The debleeded hexaphonic pickup: one 6-channel wav per track, one string per
# channel, with crosstalk between the pickup's elements suppressed. This is the
# whole point of the dataset — a monophonic tracker on an isolated string is a
# far easier problem than the polyphonic mic mix, which is why its pitch
# estimates are worth trusting over the mix-derived ones.
_HEX_ARCHIVE = "audio_hex-pickup_debleeded.zip"
_HEX_URL     = f"{_ZENODO}/{_HEX_ARCHIVE}"

# Where the archive extracts to, and the suffix its members carry.
HEX_DIRNAME = "audio_hex-pickup_debleeded"
HEX_SUFFIX  = "_hex_cln.wav"


def download(raw_dir: Path, progress: bool = True) -> None:
    """Download GuitarSet's annotations, mic audio, and hexaphonic audio.

    Parameters
    ----------
    raw_dir:
        Destination root. Files land under raw_dir/guitarset/ — the same place
        the guitarset module puts them, deliberately.
    progress:
        Show a per-file download progress bar (measured in bytes). Enabled by
        default; pass False (or --no-progress-bar on the CLI) to silence it.
    """
    # Annotations and mic audio are exactly what guitarset needs, so reuse its
    # download rather than restating the URLs: one place to fix if the record
    # moves. Both calls skip archives that are already present.
    download_guitarset(raw_dir, progress=progress)

    dest = Path(raw_dir) / "guitarset"
    archive = dest / _HEX_ARCHIVE
    if not archive.exists():
        tmp = archive.with_suffix(archive.suffix + ".part")
        _download_file(_HEX_URL, tmp, _HEX_ARCHIVE, progress)
        tmp.replace(archive)

    out_dir = dest / HEX_DIRNAME
    if not out_dir.exists():
        print(f"Extracting {_HEX_ARCHIVE} …")
        extract_and_rename_sharp(archive, out_dir)
