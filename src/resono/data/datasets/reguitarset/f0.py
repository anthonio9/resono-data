"""FCNF0++ pitch tracking over the hexaphonic channels.

This is the expensive half of the pipeline and it is deliberately a separate
command from preprocessing. The merge heuristics downstream want many
iterations; re-running a neural tracker over eighteen hours of audio for each
one is not viable, so the raw estimates are cached and preprocess reads them.

The tracker is stock FCNF0++ from upstream ``penn``. With ``checkpoint=None``
penn fetches the released weights from HuggingFace on first use, so there is no
checkpoint path to keep in sync.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from resono.data.datasets.reguitarset.download import HEX_DIRNAME
from resono.data.datasets.reguitarset.hex import (
    HEX_SUFFIX,
    N_STRINGS,
    load_hex,
    string_frequency_bounds,
)

# GuitarSet's native analysis grid: hop 256 at 44.1 kHz. reguitarset works on
# this grid rather than inheriting guitarset's 22050/256 default, which is
# 11.6 ms — half the resolution the annotations actually carry.
NATIVE_SAMPLE_RATE = 11025
NATIVE_HOP_SIZE    = 64

CONFIG_FILENAME = "config.json"


def track_f0(
    raw_dir: Path,
    f0_dir: Path,
    sample_rate: int = NATIVE_SAMPLE_RATE,
    hop_size: int = NATIVE_HOP_SIZE,
    batch_size: int = 128,
    gpu: int | None = None,
    limit: int | None = None,
    workers: int = 3,
    overwrite: bool = False,
    progress: bool = True,
) -> None:
    """Estimate per-string F0 and periodicity, and cache them.

    Produces, in f0_dir:
        {stem}-f0.npy           float32 (6, n_frames)  Hz
        {stem}-periodicity.npy  float32 (6, n_frames)  0..1
        config.json             the grid these were computed on

    Parameters
    ----------
    sample_rate, hop_size:
        Define the output frame grid, which must match the cache the labels
        will be written to: frame f is centred at f * hop_size / sample_rate
        seconds. Defaults are GuitarSet's native 5.805 ms.
    batch_size:
        Frames per forward pass. Bounds peak memory and does not affect
        results. Measured on one 22 s channel: runtime is flat across
        128/512/2048 (56.7/56.5/56.8 s) while peak RSS scales linearly
        (0.67/1.38/4.21 GB). Large batches therefore buy nothing and cost the
        headroom that makes ``workers`` possible, so the default is small.
    limit:
        Process only the first N tracks. Use this to measure throughput before
        committing to a full run — inference is per-frame, so cost scales
        linearly with track count and inversely with hop size.
    workers:
        Tracks to process concurrently, each pinned to one torch thread.
        The forward pass does not get faster with more torch threads
        (measured: 30.4-30.8 s at 1, 4 and 8), so the only parallelism
        available is across tracks — and it saturates quickly, because the
        work is memory-bound rather than compute-bound: 36.9 s per channel
        sequentially, 21.8 s at 3 workers, 21.2 s at 6. Default 3 takes
        nearly all of the available gain for ~2 GB. ``0`` means one per CPU,
        which on a small machine will thrash rather than help.
    overwrite:
        Recompute tracks that already have output. Off by default, which makes
        the command resumable: an interrupted run continues where it stopped.
    """
    from tqdm import tqdm

    hex_root = Path(raw_dir) / "guitarset" / HEX_DIRNAME
    paths = {
        p.name[: -len(HEX_SUFFIX)]: p for p in hex_root.rglob(f"*{HEX_SUFFIX}")
    }
    if not paths:
        raise FileNotFoundError(
            f"No *{HEX_SUFFIX} under {hex_root}. "
            "Has 'reguitarset download' been run?"
        )
    stems = sorted(paths)
    if limit is not None:
        stems = stems[:limit]

    out_dir = Path(f0_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds = string_frequency_bounds()
    _write_config(out_dir, sample_rate, hop_size, bounds)

    if not overwrite:
        done = [s for s in stems if _is_done(out_dir, s)]
        stems = [s for s in stems if s not in set(done)]
        if done:
            print(f"Skipping {len(done)} tracks already in {out_dir}")
    if not stems:
        print("Nothing to do — every track is already tracked.")
        return

    if workers == 0:
        workers = os.cpu_count() or 1
    jobs = [
        (paths[stem], out_dir, sample_rate, hop_size, batch_size, bounds, gpu,
         1 if workers > 1 else torch_threads())
        for stem in stems
    ]

    if workers > 1:
        # 'spawn' rather than fork: torch and its thread pools do not survive
        # being forked mid-flight, and a worker that deadlocks on import is
        # far harder to diagnose than the second of interpreter startup this
        # costs against a job measured in hours.
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn")
        ) as pool:
            for _ in tqdm(
                pool.map(_track_one, jobs), total=len(jobs),
                desc="Tracking F0", unit="track", disable=not progress,
            ):
                pass
    else:
        for job in tqdm(
            jobs, desc="Tracking F0", unit="track", disable=not progress
        ):
            _track_one(job)

    print(f"Tracked {len(stems)} tracks → {out_dir}")


def _is_done(out_dir: Path, stem: str) -> bool:
    """Has this track already been tracked?

    Both files must exist: a run killed between the two saves would otherwise
    look complete and leave the periodicity missing.
    """
    return (
        (out_dir / f"{stem}-f0.npy").exists()
        and (out_dir / f"{stem}-periodicity.npy").exists()
    )


def torch_threads() -> int:
    """Torch's own default thread count, for the single-worker path."""
    import torch

    return torch.get_num_threads()


def _track_one(job) -> str:
    """Track one file's six channels and save them.

    Module level, and taking a single plain tuple, because 'spawn' has to
    pickle both this function and its argument.
    """
    import penn
    import torch

    (path, out_dir, sample_rate, hop_size, batch_size, bounds, gpu, threads) = job
    torch.set_num_threads(threads)

    stem = path.name[: -len(HEX_SUFFIX)]
    audio, sr = load_hex(path)
    hop_seconds = hop_size / sample_rate

    # The frame count the cache will have. penn resamples internally, so
    # feed it the native audio and reconcile lengths against this.
    n_frames = int(len(audio[0]) * sample_rate / sr) // hop_size

    f0 = np.zeros((N_STRINGS, n_frames), dtype=np.float32)
    periodicity = np.zeros((N_STRINGS, n_frames), dtype=np.float32)

    # Channel c carries string c. Nothing upstream documents this — it was
    # established by measurement; see 'String ordering' in README.adoc for
    # the evidence. It is the assumption the whole dataset rests on, since
    # relabel.py then rewrites string s from row s.
    for s, channel in enumerate(audio):
        fmin, fmax = bounds[s]
        pitch, period = penn.from_audio(
            torch.from_numpy(channel)[None],
            sample_rate=sr,
            hopsize=hop_seconds,
            fmin=fmin,
            fmax=fmax,
            checkpoint=None,
            batch_size=batch_size,
            # 'zero' pads half a window on both sides, putting frame i's
            # centre at exactly i * hopsize. penn's default 'half-window'
            # pads nothing, which shifts every estimate late by half a
            # window — 64 ms — against the grid the labels sit on.
            center="zero",
            # penn 1.0.0 already defaults to this, but the relabelling
            # downstream leans on it: with Viterbi doing the smoothing,
            # relabel.py only has to fold octaves, and applies no
            # smoothing of its own. Stated explicitly so a future change
            # to penn's default cannot quietly remove that step.
            decoder="viterbi",
            gpu=gpu,
        )
        f0[s]          = _fit(pitch[0].cpu().numpy(), n_frames)
        periodicity[s] = _fit(period[0].cpu().numpy(), n_frames)

    # Periodicity first, so _is_done() cannot see a half-written track: it
    # requires the f0 file, which is written last.
    np.save(out_dir / f"{stem}-periodicity.npy", periodicity)
    np.save(out_dir / f"{stem}-f0.npy", f0)
    return stem


def _fit(values: np.ndarray, n_frames: int) -> np.ndarray:
    """Force penn's output onto exactly n_frames.

    penn derives its own frame count from the padding mode and the resampled
    length, which can land a frame either side of the cache's
    ``n_samples // hop_size``. Reconciling here — rather than trusting the two
    to agree — keeps every array in the cache the same length.
    """
    if len(values) >= n_frames:
        return values[:n_frames].astype(np.float32)
    padded = np.zeros(n_frames, dtype=np.float32)
    padded[: len(values)] = values
    return padded


# ---------------------------------------------------------------------------
# Grid bookkeeping
# ---------------------------------------------------------------------------

def _write_config(
    out_dir: Path,
    sample_rate: int,
    hop_size: int,
    bounds: list[tuple[float, float]],
) -> None:
    with open(out_dir / CONFIG_FILENAME, "w") as f:
        json.dump(
            {
                "sample_rate": sample_rate,
                "hop_size": hop_size,
                "hop_seconds": hop_size / sample_rate,
                "center": "zero",
                "tracker": "fcnf0++",
                "fmin_fmax_per_string": [list(b) for b in bounds],
            },
            f,
            indent=2,
        )


def read_config(f0_dir: Path) -> dict:
    """Load the grid an F0 cache was computed on."""
    path = Path(f0_dir) / CONFIG_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — {f0_dir} is not an F0 cache. "
            "Run 'reguitarset track-f0' first."
        )
    with open(path) as f:
        return json.load(f)


def load_f0(
    f0_dir: Path, stem: str, sample_rate: int, hop_size: int, n_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load one track's F0 and periodicity, resampled onto the target grid.

    An F0 cache is only meaningful on the grid it was computed on, so the
    caller's grid is checked rather than assumed. Where the target hop is an
    exact integer multiple of the cached one — 11.6 ms against the native
    5.8 ms, say — the coarser grid is served by decimation, which is exact and
    avoids interpolating across any octave jump the tracker left behind. Any
    other combination is refused: silently interpolating a pitch contour onto a
    grid it was not computed for is how a systematic timing error gets in.
    """
    config = read_config(f0_dir)
    cached_hop_seconds = config["hop_seconds"]
    target_hop_seconds = hop_size / sample_rate

    ratio = target_hop_seconds / cached_hop_seconds
    stride = int(round(ratio))
    if stride < 1 or abs(ratio - stride) > 1e-9:
        raise ValueError(
            f"F0 cache in {f0_dir} is on a {cached_hop_seconds * 1000:.4f} ms grid; "
            f"target is {target_hop_seconds * 1000:.4f} ms, which is not an integer "
            "multiple. Re-run 'reguitarset track-f0' with matching "
            "--sample-rate/--hop-size."
        )

    root = Path(f0_dir)
    f0 = np.load(root / f"{stem}-f0.npy")[:, ::stride]
    periodicity = np.load(root / f"{stem}-periodicity.npy")[:, ::stride]

    return _fit_2d(f0, n_frames), _fit_2d(periodicity, n_frames)


def _fit_2d(values: np.ndarray, n_frames: int) -> np.ndarray:
    """Trim or zero-pad a (6, n) array to exactly n_frames columns."""
    if values.shape[-1] >= n_frames:
        return values[:, :n_frames]
    padded = np.zeros((values.shape[0], n_frames), dtype=values.dtype)
    padded[:, : values.shape[-1]] = values
    return padded
