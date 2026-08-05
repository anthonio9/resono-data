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
    batch_size: int = 2048,
    gpu: int | None = None,
    limit: int | None = None,
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
        Frames per forward pass. Bounds peak memory; does not affect results.
    limit:
        Process only the first N tracks. Use this to measure throughput before
        committing to a full run — inference is per-frame, so cost scales
        linearly with track count and inversely with hop size.
    """
    import penn
    import torch
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
    hop_seconds = hop_size / sample_rate
    _write_config(out_dir, sample_rate, hop_size, bounds)

    for stem in tqdm(stems, desc="Tracking F0", unit="track", disable=not progress):
        audio, sr = load_hex(paths[stem])

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

        np.save(out_dir / f"{stem}-f0.npy", f0)
        np.save(out_dir / f"{stem}-periodicity.npy", periodicity)

    print(f"Tracked {len(hex_files)} tracks → {out_dir}")


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
