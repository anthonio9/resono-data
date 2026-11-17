"""Measure and correct label timing for one cached dataset.

measure  estimates each track's label shift and jitter (snapper.estimate_prior)
         and the dataset's prior, the median over tracks that had enough
         lone notes to measure.
apply    writes a corrected copy of the dataset under another cache root:
           constant   every track moved by the same number of frames
           per-track  every track moved by its own measured shift, rounded
                      to frames (the dataset prior where it couldn't be
                      measured)
           snap       per-track, then each snapped note's start moved onto
                      the frame holding its attack; the note's end stays
         Audio and any other files are hard-linked (copied across devices).
         alignment.json records what was done; a dataset that already has
         one is refused, so a correction is never applied twice.
"""
import json
import multiprocessing
import os
import shutil
from pathlib import Path

import numpy as np

from resono.data.align import snapper
from resono.data.align.labels import move_note_start, notes_of, translate
from resono.data.align.snapper import Prior

LABELS = ("onset", "voiced", "pitch")
MODES = ("constant", "per-track", "snap")
RECORD = "alignment.json"


def stems_of(dataset_dir: Path) -> list[str]:
    """Stems that carry audio and all three label arrays."""
    stems = sorted(p.name[: -len("-onset.npy")] for p in dataset_dir.glob("*-onset.npy"))
    return [s for s in stems
            if all((dataset_dir / f"{s}-{kind}.npy").exists() for kind in ("audio",) + LABELS)]


def _load(dataset_dir: Path, stem: str):
    return {kind: np.load(dataset_dir / f"{stem}-{kind}.npy") for kind in LABELS}


def _measure_track(args):
    dataset_dir, stem, sample_rate, hop_size = args
    labels = _load(dataset_dir, stem)
    audio = np.load(dataset_dir / f"{stem}-audio.npy", mmap_mode="r")
    notes = notes_of(labels["onset"], labels["pitch"])
    return stem, snapper.estimate_prior(audio, sample_rate, hop_size, notes)


def _snap_track(args):
    dataset_dir, stem, sample_rate, hop_size, prior = args
    labels = _load(dataset_dir, stem)
    audio = np.load(dataset_dir / f"{stem}-audio.npy", mmap_mode="r")
    notes = notes_of(labels["onset"], labels["pitch"])
    return stem, notes, snapper.snap_notes(audio, sample_rate, hop_size, notes, prior)


def _pool(workers: int):
    # Workers are spawned, not forked, so these limits reach their BLAS at
    # import: eight processes each running a full thread pool on small
    # matrix products made the snapper several times slower.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    return multiprocessing.get_context("spawn").Pool(workers)


def measure(
    dataset_dir: Path, sample_rate: int, hop_size: int, workers: int | None = None,
) -> tuple[dict[str, Prior | None], Prior | None]:
    """Per-track priors (None where too few lone notes) and the dataset prior."""
    stems = stems_of(dataset_dir)
    if not stems:
        raise FileNotFoundError(f"No cached tracks with onset labels in {dataset_dir}")
    with _pool(workers or os.cpu_count()) as pool:
        priors = dict(pool.map(_measure_track,
                               [(dataset_dir, s, sample_rate, hop_size) for s in stems]))
    measured = [p for p in priors.values() if p is not None]
    dataset_prior = None
    if measured:
        dataset_prior = Prior(float(np.median([p.shift_ms for p in measured])),
                              float(np.median([p.jitter_ms for p in measured])), 0)
    return priors, dataset_prior


def summary(priors: dict[str, Prior | None], dataset_prior: Prior | None,
            sample_rate: int, hop_size: int) -> str:
    frame_ms = hop_size / sample_rate * 1000
    measured = [p for p in priors.values() if p is not None]
    lines = [f"{len(priors)} tracks, {frame_ms:.2f} ms frames; "
             f"{len(measured)} with enough lone notes to measure "
             f"(at least {snapper.MIN_ANCHORS})"]
    if not measured:
        lines.append("no prior: use apply --mode constant with a shift of your choice")
        return "\n".join(lines)
    shift = np.percentile([p.shift_ms for p in measured], [10, 50, 90])
    jitter = np.percentile([p.jitter_ms for p in measured], [10, 50, 90])
    lines += [
        "attack minus label, per track: p10 {:+.1f}  median {:+.1f}  p90 {:+.1f} ms".format(*shift),
        "jitter, per track:             p10 {:.1f}  median {:.1f}  p90 {:.1f} ms".format(*jitter),
        f"dataset prior: shift {dataset_prior.shift_ms:+.1f} ms, jitter {dataset_prior.jitter_ms:.1f} ms"
        f" -> constant correction {round(dataset_prior.shift_ms / frame_ms):+d} frames",
    ]
    return "\n".join(lines)


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def apply(
    dataset_dir: Path, output_dir: Path, sample_rate: int, hop_size: int, mode: str,
    shift_frames: int = 0, workers: int | None = None, force: bool = False,
) -> dict:
    """Write the corrected dataset to output_dir (a dataset directory); -> the record."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    if dataset_dir.resolve() == output_dir.resolve():
        raise ValueError("output must be a different directory from the source")
    if (dataset_dir / RECORD).exists() and not force:
        raise ValueError(f"{dataset_dir} was already aligned ({RECORD}); "
                         f"use force to align it again")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty")
    stems = stems_of(dataset_dir)
    if not stems:
        raise FileNotFoundError(f"No cached tracks with onset labels in {dataset_dir}")
    frame_ms = hop_size / sample_rate * 1000

    priors, dataset_prior, snaps = {}, None, {}
    if mode != "constant":
        priors, dataset_prior = measure(dataset_dir, sample_rate, hop_size, workers)
        if dataset_prior is None:
            raise ValueError(f"no track in {dataset_dir} had enough lone notes to "
                             f"measure; use mode 'constant'")
    if mode == "snap":
        with _pool(workers or os.cpu_count()) as pool:
            jobs = [(dataset_dir, s, sample_rate, hop_size, priors[s] or dataset_prior)
                    for s in stems]
            snaps = {stem: (notes, result) for stem, notes, result in pool.map(_snap_track, jobs)}

    output_dir.mkdir(parents=True, exist_ok=True)
    record = {"source": str(dataset_dir.resolve()), "mode": mode,
              "sample_rate": sample_rate, "hop_size": hop_size,
              "dataset_prior": dataset_prior._asdict() if dataset_prior else None,
              "tracks": {}}
    for stem in stems:
        prior = priors.get(stem)
        if mode == "constant":
            frames = shift_frames
        else:
            frames = round((prior or dataset_prior).shift_ms / frame_ms)
        labels = _load(dataset_dir, stem)
        onset, voiced, pitch = translate(labels["onset"], labels["voiced"], labels["pitch"],
                                         frames)
        entry = {"shift_frames": frames,
                 "prior": prior._asdict() if prior else None}
        if mode == "snap":
            notes, result = snaps[stem]
            status, moved = {}, 0
            # Later notes first: moving a start earlier may shorten the note
            # before it, never one after it.
            for note, snap in sorted(zip(notes, result), key=lambda x: -x[0].frame):
                status[snap.status] = status.get(snap.status, 0) + 1
                if snap.status in snapper.SNAPPED:
                    moved += move_note_start(onset, voiced, pitch, note.string,
                                             note.frame + frames, snap.sample // hop_size)
            entry.update(status=status, moved=moved, notes=len(notes))
        record["tracks"][stem] = entry
        for kind, array in (("onset", onset), ("voiced", voiced), ("pitch", pitch)):
            np.save(output_dir / f"{stem}-{kind}.npy", array)

    label_files = {f"{s}-{kind}.npy" for s in stems for kind in LABELS}
    for source in dataset_dir.iterdir():
        if source.is_file() and source.name not in label_files and source.name != RECORD:
            _link_or_copy(source, output_dir / source.name)
    with open(output_dir / RECORD, "w") as f:
        json.dump(record, f, indent=1)
    return record


def apply_summary(record: dict) -> str:
    tracks = record["tracks"].values()
    frames = [t["shift_frames"] for t in tracks]
    values, counts = np.unique(frames, return_counts=True)
    lines = [f"{len(frames)} tracks, mode {record['mode']}; frames moved per track: "
             + ", ".join(f"{v:+d} x{c}" for v, c in zip(values, counts))]
    if record["mode"] == "snap":
        status = {}
        for t in tracks:
            for k, v in t["status"].items():
                status[k] = status.get(k, 0) + v
        notes = sum(t["notes"] for t in tracks)
        moved = sum(t["moved"] for t in tracks)
        lines.append("snaps: " + ", ".join(f"{k} {v / notes:.0%}" for k, v in sorted(status.items()))
                     + f"; starts moved beyond the track shift: {moved / max(notes, 1):.0%}")
    return "\n".join(lines)
