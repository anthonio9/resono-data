"""Tests for flattening the pitch drop at the end of a note.

Each case builds one string's labels directly on the frame grid, so the
behaviour under test is visible in the fixture. The grid is GuitarSet's native
5.805 ms, which puts the duration thresholds between frame counts: a decay must
be under 60 ms (10.3 frames), a gesture at least 100 ms (17.2 frames).
"""
import numpy as np
import pytest

from resono.data.datasets.guitarset.tails import (
    classify_drop,
    detect_drop,
    flatten_tails,
)

N_STRINGS = 6
HOP_SECONDS = 64 / 11025


def _blank(n_frames):
    return (
        np.zeros((N_STRINGS, n_frames), dtype=np.float32),
        np.full((N_STRINGS, n_frames), -1, dtype=np.int32),
    )


def _note(pitch, note_ids, string, note, start, freqs):
    frames = np.arange(start, start + len(freqs))
    pitch[string, frames] = freqs
    note_ids[string, frames] = note
    return frames


def _sliding(start_hz, end_hz, n):
    """n frames moving from start_hz to end_hz in log-frequency."""
    return (start_hz * 2 ** np.linspace(0, np.log2(end_hz / start_hz), n)).astype(
        np.float32
    )


def _one_note(departure, n_body=40, n_frames=200):
    pitch, ids = _blank(n_frames)
    freqs = np.concatenate([np.full(n_body, 110.0, dtype=np.float32), departure])
    frames = _note(pitch, ids, 0, 0, 0, freqs)
    return pitch, ids, frames


# ---------------------------------------------------------------------------
# The decay case: what this exists to remove
# ---------------------------------------------------------------------------

def test_short_drop_is_flattened():
    """A 29 ms fall cannot be a gesture, so it is held flat."""
    pitch, ids, frames = _one_note(_sliding(105.0, 97.0, 5))
    out, counts = flatten_tails(pitch, ids, HOP_SECONDS)

    assert counts["flattened"] == 1
    assert np.allclose(out[0, frames[-5:]], 110.0, rtol=1e-4)


def test_flattening_leaves_the_rest_of_the_note_alone():
    pitch, ids, frames = _one_note(_sliding(105.0, 97.0, 5))
    body = pitch[0, frames[:40]].copy()
    out, _ = flatten_tails(pitch, ids, HOP_SECONDS)
    assert np.array_equal(out[0, frames[:40]], body)


def test_input_is_not_modified():
    pitch, ids, frames = _one_note(_sliding(105.0, 97.0, 5))
    original = pitch.copy()
    flatten_tails(pitch, ids, HOP_SECONDS)
    assert np.array_equal(pitch, original)


def test_held_value_continues_the_note_rather_than_its_median():
    """A note that drifted holds the drifted value, not the note average."""
    pitch, ids = _blank(200)
    body = _sliding(110.0, 116.0, 40)
    frames = _note(pitch, ids, 0, 0, 0,
                   np.concatenate([body, _sliding(108.0, 100.0, 5)]))

    out, counts = flatten_tails(pitch, ids, HOP_SECONDS)
    assert counts["flattened"] == 1

    held = out[0, frames[-5:]]
    assert np.ptp(held) == pytest.approx(0.0, abs=1e-3)          # constant
    assert held[0] == pytest.approx(float(np.median(body[-5:])), rel=2e-3)


# ---------------------------------------------------------------------------
# The gestures that must survive
# ---------------------------------------------------------------------------

def test_settling_descent_is_preserved():
    """A 116 ms fall that settles is a slide or released bend, not decay."""
    departure = np.concatenate([
        _sliding(108.0, 98.0, 10),
        np.full(10, 98.0, dtype=np.float32),      # arrives and stays
    ])
    pitch, ids, frames = _one_note(departure)
    out, counts = flatten_tails(pitch, ids, HOP_SECONDS)

    assert counts["preserved"] == 1
    assert np.allclose(out[0, frames[-10:]], 98.0, rtol=1e-4)


def test_long_fall_that_never_settles_is_not_a_gesture():
    """A note dying slowly keeps falling; a gesture arrives and holds."""
    pitch, ids, _ = _one_note(_sliding(108.0, 80.0, 20))
    _, counts = flatten_tails(pitch, ids, HOP_SECONDS)
    assert counts["ambiguous-flattened"] == 1


def test_ambiguous_duration_is_flattened_but_labelled():
    """75 ms is too long for decay, too short for a finger — flag it."""
    departure = np.concatenate([
        _sliding(105.0, 98.0, 6),
        np.full(7, 98.0, dtype=np.float32),
    ])
    pitch, ids, frames = _one_note(departure)
    out, counts = flatten_tails(pitch, ids, HOP_SECONDS)

    assert counts["ambiguous-flattened"] == 1
    # Flattened anyway: writing a spurious bend is the worse error here.
    assert np.allclose(out[0, frames[-13:]], 110.0, rtol=1e-4)


# ---------------------------------------------------------------------------
# Downward only
# ---------------------------------------------------------------------------

def test_a_sharp_ending_is_left_alone():
    """Losing tension cannot raise the pitch, so a rise is not this."""
    pitch, ids, frames = _one_note(_sliding(113.0, 125.0, 5))
    original = pitch[0, frames].copy()
    out, counts = flatten_tails(pitch, ids, HOP_SECONDS)

    assert counts["none"] == 1
    assert np.array_equal(out[0, frames], original)


def test_detect_drop_ignores_upward_departures():
    contour = np.full(30, 110.0)
    contour[-5:] = np.array([113.0, 116.0, 119.0, 122.0, 125.0])
    assert detect_drop(contour, 110.0) == 0


def test_body_drifting_upward_is_not_read_as_a_drop():
    """The body here spans ~92 cents, so both ends sit far from its median.

    An absolute-value test would read most of the note as departure; a signed
    one sees only the five falling frames.
    """
    pitch, ids = _blank(200)
    body = _sliding(110.0, 116.0, 40)
    _note(pitch, ids, 0, 0, 0, np.concatenate([body, _sliding(108.0, 100.0, 5)]))

    contour = pitch[0, :45]
    assert detect_drop(contour, float(np.median(contour))) == 5


# ---------------------------------------------------------------------------
# Note boundaries and no-op cases
# ---------------------------------------------------------------------------

def test_note_that_ends_on_pitch_is_untouched():
    pitch, ids = _blank(100)
    frames = _note(pitch, ids, 0, 0, 0, np.full(50, 110.0, dtype=np.float32))
    original = pitch[0, frames].copy()

    out, counts = flatten_tails(pitch, ids, HOP_SECONDS)
    assert counts["none"] == 1
    assert np.array_equal(out[0, frames], original)


def test_adjacent_notes_are_flattened_independently():
    """Two notes touching with no unvoiced gap keep separate references.

    A contiguous-run segmentation would merge them and take a reference
    polluted by the other note's pitch — the case note_ids exists to prevent.
    """
    pitch, ids = _blank(200)
    first  = _note(pitch, ids, 0, 0, 0,
                   np.concatenate([np.full(40, 110.0, dtype=np.float32),
                                   _sliding(105.0, 98.0, 5)]))
    second = _note(pitch, ids, 0, 1, 45,
                   np.concatenate([np.full(40, 220.0, dtype=np.float32),
                                   _sliding(210.0, 196.0, 5)]))

    out, counts = flatten_tails(pitch, ids, HOP_SECONDS)
    assert counts["flattened"] == 2
    assert np.allclose(out[0, first[-5:]], 110.0, rtol=1e-4)
    assert np.allclose(out[0, second[-5:]], 220.0, rtol=1e-4)


def test_every_string_is_processed():
    pitch, ids = _blank(200)
    for s in range(N_STRINGS):
        _note(pitch, ids, s, 0, 0,
              np.concatenate([np.full(40, 110.0, dtype=np.float32),
                              _sliding(105.0, 97.0, 5)]))
    out, counts = flatten_tails(pitch, ids, HOP_SECONDS)
    assert counts["flattened"] == N_STRINGS
    assert np.allclose(out[:, 40:45], 110.0, rtol=1e-4)


def test_silent_string_contributes_nothing():
    pitch, ids = _blank(100)
    out, counts = flatten_tails(pitch, ids, HOP_SECONDS)
    assert sum(counts.values()) == 0
    assert not out.any()


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def test_detect_drop_finds_only_the_trailing_run():
    """A dip in the middle of a note is not a departure at its end."""
    contour = np.full(30, 110.0)
    contour[10:13] = 100.0
    contour[-4:] = np.array([104.0, 101.0, 98.0, 95.0])
    assert detect_drop(contour, 110.0) == 4


def test_detect_drop_ignores_departures_under_the_threshold():
    contour = np.full(30, 110.0)
    contour[-5:] = 110.0 * 2 ** (-10 / 1200)          # 10 cents flat
    assert detect_drop(contour, 110.0, threshold_cents=25.0) == 0


@pytest.mark.parametrize(
    "duration_ms, settled, expected",
    [
        (23.0, True,  "flattened"),                   # the measured median
        (59.0, False, "flattened"),
        (75.0, True,  "ambiguous-flattened"),
        (120.0, True, "preserved"),
        (120.0, False, "ambiguous-flattened"),
        (0.0,  False, "none"),
    ],
)
def test_classify_drop(duration_ms, settled, expected):
    tail = np.full(5, -200.0) if settled else np.array([-50.0, -120.0, -200.0])
    assert classify_drop(duration_ms, tail) == expected
