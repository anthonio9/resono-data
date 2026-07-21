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
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

GAPS_DIRNAME = "gaps"

_REPO_ID = "xavriley/GAPS"
_METADATA = "gaps_metadata_with_splits.csv"

# Annotation directories and their file extensions. Audio is handled apart
# because it is ~11 GB against a few megabytes for everything else.
_ANNOTATION_DIRS = {
    "musicxml": ".xml",
    "midi": ".mid",
    "syncpoints": ".json",
}


def download(
    raw_dir: Path,
    audio: bool = True,
    progress: bool = True,
    max_workers: int = 8,
) -> None:
    """Download GAPS into raw_dir/gaps/.

    Only the 300 tracks carrying an official split are fetched. The other 101
    rows of the metadata are scores the authors' own alignment check rejected,
    and their annotations are not trustworthy.

    Fetching goes through ``huggingface_hub`` rather than plain HTTP, which
    matters mostly for the audio: it resumes partial transfers, verifies what
    it already has instead of re-fetching, and downloads in parallel. Over
    11 GB a dropped connection would otherwise restart a whole file.

    Parameters
    ----------
    audio:
        Fetch the audio as well as the annotations. Roughly 11 GB across 300
        files. Pass False (or --no-audio on the CLI) for the annotations
        alone, which is enough to exercise everything except preprocessing.
    progress:
        Show download progress bars. Enabled by default; pass False (or
        --no-progress-bar on the CLI) to silence it.
    max_workers:
        Parallel download threads.
    """
    dest = Path(raw_dir) / GAPS_DIRNAME
    dest.mkdir(parents=True, exist_ok=True)

    (enable_progress_bars if progress else disable_progress_bars)()

    # The metadata names the tracks, so it has to arrive first.
    hf_hub_download(
        repo_id=_REPO_ID,
        repo_type="dataset",
        filename=_METADATA,
        local_dir=dest,
    )

    track_ids = read_track_ids(dest)
    if not track_ids:
        raise ValueError(f"{dest / _METADATA} lists no tracks with a split assigned")

    wanted = dict(_ANNOTATION_DIRS)
    if audio:
        wanted["audio"] = ".wav"

    snapshot_download(
        repo_id=_REPO_ID,
        repo_type="dataset",
        local_dir=dest,
        max_workers=max_workers,
        # Exact paths rather than globs: the repo holds 403 tracks and only
        # the 300 split ones are wanted.
        allow_patterns=[
            f"{directory}/{track_id}{suffix}"
            for track_id in track_ids
            for directory, suffix in wanted.items()
        ],
    )

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
