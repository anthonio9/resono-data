"""Hexaphonic pickup audio: locating and loading it.

GuitarSet's debleeded hex audio is one 6-channel wav per track, channel c
carrying string c, low E first — the same order as the JAMS ``pitch_contour``
annotations and as resono's own string axis. That ordering is stated nowhere:
not in the filenames, the Zenodo record, the GuitarSet site, the ISMIR paper,
or the JAMS metadata. It was established by measurement instead — see the
'String ordering' section of README.adoc for the evidence and the method.
"""
from pathlib import Path

import numpy as np
import soundfile as sf

from resono.data.datasets.reguitarset.download import HEX_DIRNAME, HEX_SUFFIX

N_STRINGS = 6

# Standard tuning, low E first — the order resono uses on its string axis
# (see the 'String (0 = low E)' axis label in resono.data.plot).
OPEN_STRING_HZ = (82.41, 110.00, 146.83, 196.00, 246.94, 329.63)


def load_hex(path: Path) -> tuple[np.ndarray, int]:
    """Read a hexaphonic wav as (6, n_samples) float32 at its native rate.

    soundfile returns interleaved (n_samples, n_channels); this transposes to
    put strings on the leading axis, matching the (6, n_frames) layout every
    label array in the cache uses.
    """
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[1] != N_STRINGS:
        raise ValueError(
            f"{path} has {audio.shape[1]} channels, expected {N_STRINGS}"
        )
    return np.ascontiguousarray(audio.T), sr


def string_frequency_bounds(
    margin_cents: float = 200.0, max_fret: int = 22
) -> list[tuple[float, float]]:
    """Per-string (fmin, fmax) in Hz, for constraining a pitch tracker.

    A tracker allowed the full 31–1984 Hz that FCNF0++ can represent will
    happily return an octave error; restricting each string to the range it can
    physically produce makes most of those errors unrepresentable, which is the
    cheapest accuracy win available here. The margin leaves room for tuning
    drift and bends rather than clipping a legitimately sharp note.
    """
    scale = 2.0 ** (margin_cents / 1200.0)
    return [
        (open_hz / scale, open_hz * 2.0 ** (max_fret / 12.0) * scale)
        for open_hz in OPEN_STRING_HZ
    ]
