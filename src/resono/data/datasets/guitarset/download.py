import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

_ZENODO = "https://zenodo.org/records/3371780/files"
_FILES = {
    "annotation.zip":     f"{_ZENODO}/annotation.zip",
    "audio_mono-mic.zip": f"{_ZENODO}/audio_mono-mic.zip",
}


def download(raw_dir: Path, progress: bool = True) -> None:
    """Download GuitarSet from Zenodo record 3371780.

    Parameters
    ----------
    raw_dir:
        Destination root; files land under raw_dir/guitarset/.
    progress:
        Show a per-file download progress bar (measured in bytes). Enabled by
        default; pass False (or --no-progress-bar on the CLI) to silence it.
    """
    dest = Path(raw_dir) / "guitarset"
    dest.mkdir(parents=True, exist_ok=True)

    for filename, url in _FILES.items():
        archive = dest / filename
        if not archive.exists():
            # Download to a temporary path and rename on success, so an
            # interrupted transfer never leaves a corrupt archive that the
            # existence check would happily skip on the next run.
            tmp = archive.with_suffix(archive.suffix + ".part")
            _download_file(url, tmp, filename, progress)
            tmp.replace(archive)

        # Extract each archive into its own named directory (audio_mono-mic/,
        # annotation/). This makes the on-disk layout deterministic regardless
        # of whether the zip wraps its contents in a top-level folder; preprocess
        # discovers files with rglob and so tolerates any nesting depth.
        out_dir = dest / filename.replace(".zip", "")
        if not out_dir.exists():
            print(f"Extracting {filename} …")
            extract_and_rename_sharp(archive, out_dir)


def extract_and_rename_sharp(archive: Path, out_dir: Path) -> None:
    """Unpack an archive, rewriting '#' to 'sharp' in member names.

    GuitarSet track IDs encode the key, and 48 of the 360 tracks are in a sharp
    key — ``00_Funk3-112-C#_comp``. That '#' is hostile to tooling: Kaggle
    rejects it in filenames outright, which is why re-hosted copies of these
    archives circulate with 'Csharp' instead. A raw tree assembled from a
    mixture of the two spellings joins across archives by filename and silently
    drops every sharp-key track, because the missing ones never appear in the
    index rather than raising.

    Applying the substitution here — at the one point that writes these files —
    makes the on-disk layout self-consistent whatever spelling an archive
    arrived with, so nothing downstream needs to know the discrepancy exists.
    """
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            target = out_dir / member.filename.replace("#", "sharp")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, open(target, "wb") as destination:
                shutil.copyfileobj(source, destination)


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
