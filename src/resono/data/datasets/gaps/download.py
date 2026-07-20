"""Download GAPS from the HuggingFace mirror.

GAPS is distributed in two places and only one of them is usable here: Zenodo
record 13962272 is annotations only, while the HuggingFace v1.1 release adds
the audio. Files there sit in parallel ``audio/ midi/ musicxml/ syncpoints/``
directories sharing one basename per track, plus a metadata CSV at the root
that carries the authors' official train/test split.

Licence: the GAPS authors distribute the dataset for non-commercial research
use only, non-transferable, and ask that it not be redistributed. The
HuggingFace mirror is tagged MIT, which conflicts with the terms published on
the project site; the stricter terms are the ones the authors state. See
https://aim-qmul.github.io/GAPS/ before using this beyond local research.
"""
import csv
import urllib.request
from pathlib import Path

from tqdm import tqdm

GAPS_DIRNAME = "gaps"

_HF_BASE = "https://huggingface.co/datasets/xavriley/GAPS/resolve/main"
_METADATA = "gaps_metadata_with_splits.csv"

# Annotation directories and their file extensions. Audio is handled apart
# because it is ~11 GB against a few megabytes for everything else.
_ANNOTATION_DIRS = {
    "musicxml": ".xml",
    "midi": ".mid",
    "syncpoints": ".json",
}


def download(raw_dir: Path, audio: bool = True, progress: bool = True) -> None:
    """Download GAPS into raw_dir/gaps/.

    Only the 300 tracks carrying an official split are fetched. The other 101
    rows of the metadata are scores the authors' own alignment check rejected,
    and their annotations are not trustworthy.

    Parameters
    ----------
    audio:
        Fetch the audio as well as the annotations. Roughly 11 GB across 300
        files. Pass False (or --no-audio on the CLI) for the annotations
        alone, which is enough to exercise everything except preprocessing.
    progress:
        Show a progress bar over files. Enabled by default; pass False (or
        --no-progress-bar on the CLI) to silence it.
    """
    dest = Path(raw_dir) / GAPS_DIRNAME
    dest.mkdir(parents=True, exist_ok=True)

    metadata = dest / _METADATA
    if not metadata.exists():
        _fetch(f"{_HF_BASE}/{_METADATA}", metadata)

    track_ids = read_track_ids(dest)
    if not track_ids:
        raise ValueError(f"{metadata} lists no tracks with a split assigned")

    wanted = dict(_ANNOTATION_DIRS)
    if audio:
        wanted["audio"] = ".wav"
    for directory in wanted:
        (dest / directory).mkdir(exist_ok=True)

    jobs = [
        (f"{_HF_BASE}/{directory}/{track_id}{suffix}",
         dest / directory / f"{track_id}{suffix}")
        for track_id in track_ids
        for directory, suffix in wanted.items()
    ]
    pending = [(url, path) for url, path in jobs if not path.exists()]

    for url, path in tqdm(
        pending, desc="Downloading", unit="file", disable=not progress
    ):
        _fetch(url, path)

    print(f"GAPS ready: {len(track_ids)} tracks → {dest}")


def read_track_ids(gaps_root: Path) -> list[str]:
    """Track ids carrying an official split, in metadata order.

    The split column is the authors' own 90:10 division by piece and is the
    authoritative track list — it is not the same set as filtering the
    metadata's f-measure column, and the two agree on only 250 of 300 tracks.
    """
    metadata = Path(gaps_root) / _METADATA
    if not metadata.exists():
        return []
    with open(metadata, newline="", encoding="utf-8") as handle:
        return [
            row["id"] for row in csv.DictReader(handle) if row.get("split", "").strip()
        ]


def read_metadata(gaps_root: Path) -> list[dict]:
    """Full metadata rows for tracks carrying an official split."""
    metadata = Path(gaps_root) / _METADATA
    if not metadata.exists():
        raise FileNotFoundError(f"{metadata} not found; run download first")
    with open(metadata, newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("split", "").strip()]


def _fetch(url: str, dest: Path) -> None:
    """Fetch url → dest via a temporary file.

    Downloading to a .part path and renaming on success means an interrupted
    transfer never leaves a truncated file that the existence check would
    happily skip on the next run.
    """
    temporary = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(dest)
