import json

import jams
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic dataset fixtures
# ---------------------------------------------------------------------------

HOP_SIZE      = 256
WINDOW_FRAMES = 8
N_STRINGS     = 6


@pytest.fixture
def fake_dataset(tmp_path):
    """Three synthetic tracks with known shapes, voiced regions, and onsets.

    Track layout (n_frames each):
        track_a : 172 frames  ~2 s at hop=256/22050 Hz
        track_b : 344 frames  ~4 s
        track_c :  86 frames  ~1 s  (valid only: 86 >= WINDOW_FRAMES)

    Voiced pattern per track: first half voiced, second half silent.
    This guarantees both voiced_idx and onset_idx are non-empty.

    Every track carries all four cache arrays, onset included — Dataset loads
    it unconditionally, so a fixture without it is not a cache the loader can
    read.
    """
    rng = np.random.default_rng(0)

    cache_dir      = tmp_path / "cache" / "guitarset"
    partitions_dir = tmp_path / "partitions"
    cache_dir.mkdir(parents=True)
    partitions_dir.mkdir()

    tracks = [
        ("track_a", 172),
        ("track_b", 344),
        ("track_c",  86),
    ]

    for stem, n_frames in tracks:
        # Deliberately non-aligned: real resampled audio almost never has a
        # length that is an exact multiple of hop_size. The extra 37 samples
        # (< HOP_SIZE, so n_frames is unchanged) exercise the boundary-padding
        # path and would expose any leakage of trailing samples into a window.
        n_samples = n_frames * HOP_SIZE + 37
        audio     = rng.standard_normal(n_samples).astype(np.float32)

        pitch  = np.zeros((N_STRINGS, n_frames), dtype=np.float32)
        voiced = np.zeros((N_STRINGS, n_frames), dtype=bool)
        onset  = np.zeros((N_STRINGS, n_frames), dtype=bool)

        # First half is voiced with realistic guitar frequencies.
        half = n_frames // 2
        freqs = [82.4, 110.0, 146.8, 196.0, 246.9, 329.6]
        for s, f in enumerate(freqs):
            pitch[s,  :half] = f
            voiced[s, :half] = True
            onset[s, 0] = True

        np.save(cache_dir / f"{stem}-audio.npy",  audio)
        np.save(cache_dir / f"{stem}-pitch.npy",  pitch)
        np.save(cache_dir / f"{stem}-voiced.npy", voiced)
        np.save(cache_dir / f"{stem}-onset.npy",  onset)

    stems = [s for s, _ in tracks]
    partition = {
        "train": stems[:2],
        "valid": [stems[2]],
        "test":  [stems[2]],
    }
    with open(partitions_dir / "guitarset.json", "w") as f:
        json.dump(partition, f)

    return {
        "cache_dir":      tmp_path / "cache",
        "partitions_dir": partitions_dir,
        "datasets":       ["guitarset"],
        "hop_size":       HOP_SIZE,
        "window_frames":  WINDOW_FRAMES,
        "tracks":         tracks,
    }


# ---------------------------------------------------------------------------
# Synthetic GuitarSet JAMS fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def guitarset_jams():
    """Minimal GuitarSet-style JAMS with 6 pitch_contour annotations.

    Each string is voiced for the first half of the 5-second clip and silent
    for the second half, creating a clear onset at frame 0 and a clear offset
    at the midpoint.
    """
    return _make_guitarset_jams(duration=5.0, hop_seconds=256 / 44100)


def _make_guitarset_jams(duration: float, hop_seconds: float) -> jams.JAMS:
    open_string_hz = [82.41, 110.00, 146.83, 196.00, 246.94, 329.63]

    jam = jams.JAMS()
    jam.file_metadata.duration = duration

    for freq in open_string_hz:
        ann = jams.Annotation(namespace="pitch_contour")
        t   = 0.0
        while t + hop_seconds <= duration:
            if t < duration / 2:
                ann.append(
                    time=t,
                    duration=hop_seconds,
                    value={"frequency": freq, "voiced": True},
                    confidence=1.0,
                )
            t += hop_seconds
        jam.annotations.append(ann)

    return jam
