"""Download Guitar-TECHS from Zenodo record 14963133.

The dataset ships as nine independently downloadable zips, one per
(player, category), totalling ~4.1 GB. They are listed separately here so a
subset can be fetched — the categories differ enough in value that pulling all
of it is often not what you want.

CC BY 4.0: freely redistributable, no request step, no non-commercial clause.
"""
import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path

from resono.data.datasets.guitarset.download import _download_file

_RECORD = "https://zenodo.org/api/records/14963133"
_ZENODO = f"{_RECORD}/files"

# Zenodo serves these through a gateway that, under load, answers with a
# 92-byte '504 Gateway Time-out' page carrying HTTP 200. urlretrieve sees a
# successful response and writes it out; the file is then a valid download of
# an invalid archive, and any existence check accepts it forever after. The
# record's own metadata publishes each file's size and MD5, so acceptance is
# gated on those rather than on the request not raising.
_RETRIES = 4
_BACKOFF_SECONDS = 60

GUITARTECHS_DIRNAME = "guitar-techs"

# Approximate sizes, for choosing a subset without hitting the network first.
ARCHIVES = {
    "P1_chords":      982,
    "P1_scales":      453,
    "P1_singlenotes": 109,
    "P1_techniques":  326,
    "P2_chords":     1151,
    "P2_scales":      471,
    "P2_singlenotes": 116,
    "P2_techniques":  396,
    "P3_music":       130,
}


def download(
    raw_dir: Path,
    archives: tuple[str, ...] | None = None,
    progress: bool = True,
) -> None:
    """Fetch and extract Guitar-TECHS archives.

    Parameters
    ----------
    raw_dir:
        Destination root; files land under raw_dir/guitar-techs/.
    archives:
        Which archives to fetch, by name. Defaults to all nine (~4.1 GB).
    progress:
        Show a per-file byte progress bar.
    """
    names = tuple(archives) if archives else tuple(ARCHIVES)
    unknown = [n for n in names if n not in ARCHIVES]
    if unknown:
        raise ValueError(
            f"unknown archive(s) {unknown}; choose from {sorted(ARCHIVES)}"
        )

    dest = Path(raw_dir) / GUITARTECHS_DIRNAME
    dest.mkdir(parents=True, exist_ok=True)

    expected = _published_checksums()

    for name in names:
        archive = dest / f"{name}.zip"
        want = expected.get(f"{name}.zip")

        if not _is_intact(archive, want):
            if archive.exists():
                print(f"{archive.name} is present but does not match the "
                      f"published checksum — refetching")
                archive.unlink()
            _fetch(f"{_ZENODO}/{name}.zip/content", archive, want, progress)

        out_dir = dest / name
        if not out_dir.exists():
            print(f"Extracting {name}.zip …")
            _extract(archive, dest)


def _published_checksums() -> dict[str, dict]:
    """Size and MD5 for every file in the record, straight from Zenodo."""
    try:
        with urllib.request.urlopen(_RECORD, timeout=30) as response:
            record = json.load(response)
    except Exception as error:                      # offline, or Zenodo down
        print(f"  warning: could not read the record metadata ({error}); "
              "falling back to archive-validity checks only")
        return {}
    return {
        f["key"]: {"size": f.get("size"), "md5": f.get("checksum", "").removeprefix("md5:")}
        for f in record.get("files", [])
    }


def _is_intact(archive: Path, want: dict | None) -> bool:
    """Is this a complete, uncorrupted copy of the published archive?

    Existence is not enough: a gateway error page saved as .zip exists, has a
    size, and is not a zip. Size is checked before the hash because it rejects
    the common failures instantly on a multi-gigabyte file.
    """
    if not archive.exists():
        return False
    if want:
        if want.get("size") and archive.stat().st_size != want["size"]:
            return False
        if want.get("md5") and _md5(archive) != want["md5"]:
            return False
        return True
    # No published metadata to compare against; fall back to asking zipfile
    # whether this is even an archive.
    try:
        with zipfile.ZipFile(archive) as zf:
            return zf.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(url: str, archive: Path, want: dict | None, progress: bool) -> None:
    """Download until the bytes match what Zenodo says they should be."""
    tmp = archive.with_suffix(".zip.part")
    for attempt in range(1, _RETRIES + 1):
        tmp.unlink(missing_ok=True)
        try:
            _download_file(url, tmp, archive.name, progress)
        except Exception as error:
            print(f"  {archive.name}: attempt {attempt} failed ({error})")
            tmp.unlink(missing_ok=True)
        else:
            if _is_intact(tmp, want):
                tmp.replace(archive)
                return
            size = tmp.stat().st_size if tmp.exists() else 0
            print(f"  {archive.name}: attempt {attempt} returned {size} bytes, "
                  "which does not match the published archive")

        if attempt < _RETRIES:
            # Zenodo asks for 60 s after a gateway timeout, and the whole
            # failure mode is load-related, so waiting is the fix.
            print(f"  retrying in {_BACKOFF_SECONDS}s …")
            time.sleep(_BACKOFF_SECONDS)

    tmp.unlink(missing_ok=True)
    raise RuntimeError(
        f"Could not download a valid {archive.name} after {_RETRIES} attempts. "
        "Zenodo may be rate-limiting; try again later or fetch fewer archives "
        "at once."
    )


def _extract(archive: Path, dest: Path) -> None:
    """Unpack, skipping the macOS resource forks the archives carry.

    Those '__MACOSX/._*' entries are not audio but do end in .wav, so anything
    globbing for media picks them up and then fails to open them.
    """
    with zipfile.ZipFile(archive) as zf:
        members = [
            n for n in zf.namelist()
            if "__MACOSX" not in n and not Path(n).name.startswith("._")
            and ".DS_Store" not in n
        ]
        zf.extractall(dest, members=members)
