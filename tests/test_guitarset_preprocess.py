import jams
import numpy as np
import pytest

from resono.data.datasets.guitarset.preprocess import (
    _remove_pitch_overhangs,
    extract_pitch_array_jams,
)

HOP_SECONDS = 256 / 44100


# ---------------------------------------------------------------------------
# Overhang-removal fixture helpers
# ---------------------------------------------------------------------------

def _make_single_note_jams(body_freq, tail_freqs, hop_seconds=HOP_SECONDS):
    """One string, one note: body frames at body_freq, tail frames at tail_freqs."""
    jam = jams.JAMS()
    jam.file_metadata.duration = (len(tail_freqs) + 20) * hop_seconds

    ann = jams.Annotation(namespace="pitch_contour")
    t = 0.0
    # body frames (index=0)
    for _ in range(20):
        ann.append(time=t, duration=hop_seconds,
                   value={"frequency": body_freq, "voiced": True, "index": 0},
                   confidence=1.0)
        t += hop_seconds
    # tail frames (same note, index=0)
    for freq in tail_freqs:
        ann.append(time=t, duration=hop_seconds,
                   value={"frequency": freq, "voiced": True, "index": 0},
                   confidence=1.0)
        t += hop_seconds
    jam.annotations.append(ann)
    return jam


# ---------------------------------------------------------------------------
# extract_pitch_array_jams
# ---------------------------------------------------------------------------

def test_output_shape(guitarset_jams):
    duration = float(guitarset_jams.file_metadata.duration)
    n_frames = int(duration / HOP_SECONDS)      # floor — matches preprocess logic
    pitch, voiced = extract_pitch_array_jams(guitarset_jams, HOP_SECONDS, n_frames)

    assert pitch.shape  == (6, n_frames)
    assert voiced.shape == (6, n_frames)


def test_output_dtypes(guitarset_jams):
    duration = float(guitarset_jams.file_metadata.duration)
    n_frames = int(duration / HOP_SECONDS)
    pitch, voiced = extract_pitch_array_jams(guitarset_jams, HOP_SECONDS, n_frames)

    assert pitch.dtype  == np.float32
    assert voiced.dtype == bool


def test_voiced_pitch_consistency(guitarset_jams):
    """Voiced frames must have non-zero pitch; unvoiced frames must be zero."""
    duration = float(guitarset_jams.file_metadata.duration)
    n_frames = int(duration / HOP_SECONDS)
    pitch, voiced = extract_pitch_array_jams(guitarset_jams, HOP_SECONDS, n_frames)

    assert (pitch[voiced]  > 0).all(),  "Voiced frame has zero pitch"
    assert (pitch[~voiced] == 0).all(), "Unvoiced frame has non-zero pitch"


def test_voiced_in_first_half_only(guitarset_jams):
    """The fixture is voiced for the first half — verify this is reflected."""
    duration = float(guitarset_jams.file_metadata.duration)
    n_frames = int(duration / HOP_SECONDS)
    _, voiced = extract_pitch_array_jams(guitarset_jams, HOP_SECONDS, n_frames)

    half = n_frames // 2
    assert voiced[:, :half].any(), "Expected voiced frames in first half"
    # Frame `half` may be voiced due to rounding at the midpoint boundary;
    # check that everything strictly after it is silent.
    assert not voiced[:, half + 1:].any(), "Expected silence after midpoint"


def test_six_strings(guitarset_jams):
    duration = float(guitarset_jams.file_metadata.duration)
    n_frames = int(duration / HOP_SECONDS)
    pitch, voiced = extract_pitch_array_jams(guitarset_jams, HOP_SECONDS, n_frames)

    assert pitch.shape[0]  == 6
    assert voiced.shape[0] == 6


# ---------------------------------------------------------------------------
# _remove_pitch_overhangs
# ---------------------------------------------------------------------------

def test_overhang_drifting_tail_silenced():
    """Tail frames that deviate > threshold must be marked unvoiced."""
    body_freq  = 220.0
    # ~120 cents above body — well over the 15-cent default threshold
    drift_freq = body_freq * (2 ** (120 / 1200))
    tail = [drift_freq] * 4   # 4 tail frames out of 24 total → last 20%

    jam = _make_single_note_jams(body_freq, tail)
    duration = float(jam.file_metadata.duration)
    n_frames  = int(duration / HOP_SECONDS)

    pitch, voiced = extract_pitch_array_jams(jam, HOP_SECONDS, n_frames)
    pitch, voiced = _remove_pitch_overhangs(
        jam, pitch, voiced, HOP_SECONDS, n_frames,
        divider=5, threshold_cents=15.0,
    )

    # String 0 tail frames must be unvoiced
    assert not voiced[0, -4:].any(), "Drifting tail frames should be silenced"


def test_overhang_stable_tail_kept():
    """Tail frames within threshold must remain voiced."""
    body_freq  = 220.0
    # Only 5 cents deviation — within the 15-cent threshold
    stable_freq = body_freq * (2 ** (5 / 1200))
    tail = [stable_freq] * 4

    jam = _make_single_note_jams(body_freq, tail)
    duration = float(jam.file_metadata.duration)
    n_frames  = int(duration / HOP_SECONDS)

    pitch, voiced = extract_pitch_array_jams(jam, HOP_SECONDS, n_frames)
    pitch, voiced = _remove_pitch_overhangs(
        jam, pitch, voiced, HOP_SECONDS, n_frames,
        divider=5, threshold_cents=15.0,
    )

    assert voiced[0, -4:].any(), "Stable tail frames should remain voiced"


def test_overhang_does_not_modify_body():
    """The note body (non-tail frames) must never be silenced."""
    body_freq  = 220.0
    drift_freq = body_freq * (2 ** (200 / 1200))
    tail = [drift_freq] * 4

    jam = _make_single_note_jams(body_freq, tail)
    duration = float(jam.file_metadata.duration)
    n_frames  = int(duration / HOP_SECONDS)

    pitch, voiced = extract_pitch_array_jams(jam, HOP_SECONDS, n_frames)
    body_voiced_before = voiced[0, :20].copy()

    pitch, voiced = _remove_pitch_overhangs(
        jam, pitch, voiced, HOP_SECONDS, n_frames,
        divider=5, threshold_cents=15.0,
    )

    assert (voiced[0, :20] == body_voiced_before).all(), "Body frames were modified"
