import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

_ZENODO = "https://zenodo.org/records/3371780/files"

# Every archive in the record that this repo can use. Keyed by the directory
# each unpacks into, which is also what preprocess globs through.
ARCHIVES = {
    "annotation":            "annotation.zip",
    "audio_mono-mic":        "audio_mono-mic.zip",
    "audio_mono-pickup_mix": "audio_mono-pickup_mix.zip",
}

# All of them by default (~1.4 GB). The microphone and the pickup are
# simultaneous recordings of the same performances under the same annotations,
# so having both on disk is what makes the two audio sources interchangeable
# without a second download.
DEFAULT_ARCHIVES = tuple(ARCHIVES)

# GuitarSet track IDs name the key, and 48 of the 360 are sharp: '00_Funk3-112-C#_comp'.
# Some redistributions replace '#' with 'sharp' because it is hostile to
# tooling — Kaggle rejects it in filenames outright — so a tree assembled from
# more than one source ends up with both spellings and any join by filename
# silently drops whatever half does not match.
_SHARP = re.compile(r"([A-G])#")


def download(
    raw_dir: Path,
    archives: tuple[str, ...] | None = None,
    rename_sharp: bool = True,
    progress: bool = True,
) -> None:
    """Download GuitarSet from Zenodo record 3371780.

    Parameters
    ----------
    raw_dir:
        Destination root; files land under raw_dir/guitarset/.
    archives:
        Which archives to fetch, by directory name; see :data:`ARCHIVES`.
        Defaults to all of them (~1.4 GB): annotations, the mono microphone
        audio, and the mono pickup mix.
    rename_sharp:
        Rewrite '#' to 'sharp' in extracted filenames, so a tree assembled
        from several sources joins on one spelling. On by default because
        mixing spellings drops tracks without raising. Turn it off to keep
        the archive's own names.
    progress:
        Show a per-file download progress bar (measured in bytes). Enabled by
        default; pass False (or --no-progress-bar on the CLI) to silence it.
    """
    names = tuple(archives) if archives else DEFAULT_ARCHIVES
    unknown = [n for n in names if n not in ARCHIVES]
    if unknown:
        raise ValueError(f"unknown archive(s) {unknown}; choose from {sorted(ARCHIVES)}")

    dest = Path(raw_dir) / "guitarset"
    dest.mkdir(parents=True, exist_ok=True)

    for name in names:
        filename = ARCHIVES[name]
        archive = dest / filename
        if not archive.exists():
            # Download to a temporary path and rename on success, so an
            # interrupted transfer never leaves a corrupt archive that the
            # existence check would happily skip on the next run.
            tmp = archive.with_suffix(archive.suffix + ".part")
            _download_file(f"{_ZENODO}/{filename}", tmp, filename, progress)
            tmp.replace(archive)

        # Extract each archive into its own named directory (audio_mono-mic/,
        # annotation/). This makes the on-disk layout deterministic regardless
        # of whether the zip wraps its contents in a top-level folder; preprocess
        # discovers files with rglob and so tolerates any nesting depth.
        out_dir = dest / name
        if not out_dir.exists():
            print(f"Extracting {filename} …")
            extract(archive, out_dir, rename_sharp=rename_sharp)


def extract(archive: Path, out_dir: Path, rename_sharp: bool = True) -> None:
    """Unpack an archive, optionally normalising sharp keys in member names.

    Applied here, at the one point that writes these files, so the on-disk
    layout is self-consistent whatever spelling an archive arrived with and
    nothing downstream has to know the discrepancy exists.
    """
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = member.filename
            if rename_sharp:
                name = _SHARP.sub(r"\1sharp", name)
            target = out_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, open(target, "wb") as destination:
                shutil.copyfileobj(source, destination)


def rename_sharp_in_place(directory: Path) -> int:
    """Rewrite '#' to 'sharp' in the names of files already on disk.

    For trees extracted before this normalisation existed, or assembled by
    hand. Returns how many files were renamed.
    """
    renamed = 0
    for path in sorted(Path(directory).rglob("*#*")):
        new = path.with_name(_SHARP.sub(r"\1sharp", path.name))
        if new != path:
            path.rename(new)
            renamed += 1
    return renamed


def _download_file(url: str, dest: Path, label: str, progress: bool) -> None:
    """Fetch url → dest, optionally driving a byte-level tqdm bar."""
    if not progress:
        print(f"Downloading {label} …")
        urllib.request.urlretrieve(url, dest)
        return

    with tqdm(
        desc=label,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
    ) as bar:
        def hook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                bar.total = total_size
            # reporthook gives cumulative block counts; update by the delta.
            bar.update(block_num * block_size - bar.n)

        urllib.request.urlretrieve(url, dest, reporthook=hook)
