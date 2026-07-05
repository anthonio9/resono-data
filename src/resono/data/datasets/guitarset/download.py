import urllib.request
import zipfile
from pathlib import Path

_ZENODO = "https://zenodo.org/records/3371780/files"
_FILES = {
    "annotation.zip":     f"{_ZENODO}/annotation.zip",
    "audio_mono-mic.zip": f"{_ZENODO}/audio_mono-mic.zip",
}


def download(raw_dir: Path) -> None:
    """Download GuitarSet from Zenodo record 3371780."""
    dest = Path(raw_dir) / "guitarset"
    dest.mkdir(parents=True, exist_ok=True)

    for filename, url in _FILES.items():
        archive = dest / filename
        if not archive.exists():
            print(f"Downloading {filename} …")
            # Download to a temporary path and rename on success, so an
            # interrupted transfer never leaves a corrupt archive that the
            # existence check would happily skip on the next run.
            tmp = archive.with_suffix(archive.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(archive)

        # Extract each archive into its own named directory (audio_mono-mic/,
        # annotation/). This makes the on-disk layout deterministic regardless
        # of whether the zip wraps its contents in a top-level folder; preprocess
        # discovers files with rglob and so tolerates any nesting depth.
        out_dir = dest / filename.replace(".zip", "")
        if not out_dir.exists():
            print(f"Extracting {filename} …")
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(out_dir)
