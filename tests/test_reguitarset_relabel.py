"""Tests for the FCNF0++ tail relabelling.

Each case builds a single string's labels directly on the frame grid — a note
body, a tail, and an FCNF0++ contour to merge in — so the behaviour under test
is visible in the fixture rather than buried in a JAMS round-trip. The
JAMS→grid path itself is already covered by test_guitarset_preprocess.
"""
import numpy as np
import pytest

from resono.data.datasets.reguitarset.relabel import (
    octave_fold,
    relabel_tails,
    summarise,
)

N_STRINGS = 6
HOP_SECONDS = 64 / 11025


def _blank(n_frames):
    return (
        np.zeros((N_STRINGS, n_frames), dtype=np.float32),
        np.zeros((N_STRINGS, n_frames), dtype=bool),
        np.full((N_STRINGS, n_frames), -1, dtype=np.int32),
    )


def _note(arrays, string, note, start, freqs):
    """Write one note onto the label arrays and return its frame indices."""
    pitch, voiced, note_ids = arrays
    frames = np.arange(start, start + len(freqs))
    pitch[string, frames] = freqs
    voiced[string, frames] = True
    note_ids[string, frames] = note
    return frames


def _f0(n_frames, string, frames, values, periodicity=1.0):
    """An FCNF0++ contour that is `values` on `frames` and silent elsewhere."""
    f0 = np.zeros((N_STRINGS, n_frames), dtype=np.float32)
    period = np.zeros((N_STRINGS, n_frames), dtype=np.float32)
    f0[string, frames] = values
    period[string, frames] = periodicity
    return f0, period


# ---------------------------------------------------------------------------
# Tail pitch: 'track'
# ---------------------------------------------------------------------------

def test_octave_jumping_tail_is_folded_back():
    """A tracker that drops an octave in the tail is corrected, not discarded.

    This is the error the whole exercise exists to fix: the pitch class is
    right and only the register is wrong, so folding recovers a usable
    estimate where dropping the frame would lose one.
    """
    n = 50
    arrays = _blank(n)
    frames = _note(arrays, 0, 0, 5, np.full(40, 110.0, dtype=np.float32))

    # FCNF0++ agrees on the body but halves in the tail — a classic octave error.
    contour = np.full(40, 110.0, dtype=np.float32)
    contour[32:] = 55.0
    f0, period = _f0(n, 0, frames, contour)

    pitch, voiced, audits = relabel_tails(
        *arrays, f0, period, stem="t", hop_seconds=HOP_SECONDS,
        tail_policy="track", offset_policy="none",
    )

    tail = frames[-8:]
    assert np.allclose(pitch[0, tail], 110.0, rtol=1e-3)
    assert audits[0].octave_folds == 8


def test_clean_tail_is_left_alone():
    """Where FCNF0++ agrees with the labels, the labels do not move."""
    n = 50
    arrays = _blank(n)
    frames = _note(arrays, 0, 0, 5, np.full(40, 196.0, dtype=np.float32))
    f0, period = _f0(n, 0, frames, np.full(40, 196.0, dtype=np.float32))

    pitch, voiced, audits = relabel_tails(
        *arrays, f0, period, stem="t", hop_seconds=HOP_SECONDS,
        tail_policy="track", offset_policy="none",
    )

    assert np.allclose(pitch[0, frames], 196.0, rtol=1e-4)
    assert audits[0].octave_folds == 0
    assert abs(audits[0].tail_cents_change_median) < 1e-6


def test_body_is_never_rewritten():
    """FCNF0++ owns the tail only; the human-corrected body is untouched.

    Given a contour that disagrees everywhere, the body must still come out
    exactly as labelled — otherwise 'surgical' is not what this does.
    """
    n = 60
    arrays = _blank(n)
    frames = _note(arrays, 2, 0, 0, np.full(50, 146.83, dtype=np.float32))
    original_body = arrays[0][2, frames[:40]].copy()

    f0, period = _f0(n, 2, frames, np.full(50, 155.0, dtype=np.float32))

    pitch, _, audits = relabel_tails(
        *arrays, f0, period, stem="t", hop_seconds=HOP_SECONDS,
        tail_policy="track", offset_policy="none",
    )

    assert np.array_equal(pitch[2, frames[:40]], original_body)
    # The disagreement is recorded rather than acted on.
    assert audits[0].body_cents_diff == pytest.approx(
        1200 * np.log2(155.0 / 146.83), abs=1.0
    )


# ---------------------------------------------------------------------------
# Tail pitch: 'hold'
# ---------------------------------------------------------------------------

def test_hold_extrapolates_a_drifting_body():
    """'hold' continues the body's own trend rather than the tracker's."""
    n = 60
    arrays = _blank(n)
    # A body drifting upward by a steady 2 cents per frame.
    body = 110.0 * 2 ** (np.arange(40) * 2.0 / 1200.0)
    frames = _note(arrays, 0, 0, 0, np.concatenate([body, np.full(10, 110.0)]))

    # FCNF0++ says something wild in the tail; 'hold' must ignore it entirely.
    contour = np.concatenate([body, np.full(10, 300.0)]).astype(np.float32)
    f0, period = _f0(n, 0, frames, contour)

    pitch, _, _ = relabel_tails(
        *arrays, f0, period, stem="t", hop_seconds=HOP_SECONDS,
        tail_policy="hold", offset_policy="none",
    )

    tail = frames[-10:]
    # Continues upward past the body's last value, and nowhere near 300 Hz.
    assert (pitch[0, tail] > body[-1]).all()
    assert (pitch[0, tail] < 130.0).all()


def test_hold_carries_a_bend_through_the_tail():
    """A bend is part of the note, so 'hold' must continue it, not cap it.

    A whole-tone bend is 200 cents and still the same note event — there is no
    new onset. Bounding the extrapolation would both flatten real playing and
    put a discontinuity at the body/tail boundary, which is why _held is
    deliberately unbounded.
    """
    n = 60
    arrays = _blank(n)
    # A whole-tone bend across the body, still rising when the body ends.
    body = 110.0 * 2 ** (np.linspace(0, 200.0, 40) / 1200.0)
    frames = _note(arrays, 0, 0, 0, np.concatenate([body, np.full(10, 123.0)]))
    f0, period = _f0(n, 0, frames, np.full(50, 123.0, dtype=np.float32))

    pitch, _, _ = relabel_tails(
        *arrays, f0, period, stem="t", hop_seconds=HOP_SECONDS,
        tail_policy="hold", offset_policy="none",
    )

    tail = pitch[0, frames[-10:]].astype(np.float64)
    body_end = float(body[-1])
    body_slope_cents = 200.0 / 39.0          # the fixture's own cents per frame

    # Continues upward from where the body left off, monotonically.
    assert (tail > body_end).all()
    assert (np.diff(tail) > 0).all()

    # No step at the seam: the first tail frame is one frame's worth of the
    # body's slope beyond it, not a jump.
    seam = 1200 * np.log2(tail[0] / body_end)
    assert seam == pytest.approx(body_slope_cents, rel=0.1)

    # Bounded by geometry rather than by a cap: a tail of 1/divider of the note
    # continues a whole-body trend by at most that share of its excursion.
    overshoot = 1200 * np.log2(tail[-1] / body_end)
    assert 0 < overshoot < 200.0 * (10 / 40) * 1.2


# ---------------------------------------------------------------------------
# Offsets
# ---------------------------------------------------------------------------

def test_trim_removes_only_the_decayed_run():
    """Trimming stops at the first frame still sounding."""
    n = 60
    arrays = _blank(n)
    frames = _note(arrays, 0, 0, 0, np.full(50, 110.0, dtype=np.float32))

    periodicity = np.full(50, 0.9, dtype=np.float32)
    periodicity[-6:] = 0.05                 # decayed run at the very end
    periodicity[20] = 0.05                  # an isolated dip mid-body
    f0, period = _f0(n, 0, frames, np.full(50, 110.0, dtype=np.float32))
    period[0, frames] = periodicity

    _, voiced, audits = relabel_tails(
        *arrays, f0, period, stem="t", hop_seconds=HOP_SECONDS,
        offset_policy="trim", periodicity_threshold=0.3,
    )

    assert audits[0].frames_trimmed == 6
    assert not voiced[0, frames[-6:]].any()
    # The mid-note dip is a tracker artefact, not the note ending twice.
    assert voiced[0, frames[20]]


def test_extend_stops_at_the_next_note():
    """A note may not grow into its successor, however long it rings."""
    n = 80
    arrays = _blank(n)
    first  = _note(arrays, 0, 0, 0,  np.full(30, 110.0, dtype=np.float32))
    second = _note(arrays, 0, 1, 40, np.full(30, 146.83, dtype=np.float32))

    # Periodicity stays high right through the gap and into the next note.
    f0 = np.zeros((N_STRINGS, n), dtype=np.float32)
    period = np.zeros((N_STRINGS, n), dtype=np.float32)
    f0[0, :70] = 110.0
    period[0, :70] = 0.9

    _, voiced, audits = relabel_tails(
        *arrays, f0, period, stem="t", hop_seconds=HOP_SECONDS,
        offset_policy="extend", periodicity_threshold=0.3,
    )

    # Frames 30..39 are free; frame 40 belongs to the second note.
    assert audits[0].frames_extended == 10
    assert voiced[0, 30:40].all()
    assert np.allclose(voiced[0, second], True)


def test_extend_respects_max_extend_frames():
    n = 200
    arrays = _blank(n)
    frames = _note(arrays, 0, 0, 0, np.full(30, 110.0, dtype=np.float32))

    f0 = np.zeros((N_STRINGS, n), dtype=np.float32)
    period = np.zeros((N_STRINGS, n), dtype=np.float32)
    f0[0, :] = 110.0
    period[0, :] = 0.9

    _, _, audits = relabel_tails(
        *arrays, f0, period, stem="t", hop_seconds=HOP_SECONDS,
        offset_policy="extend", max_extend_frames=12,
    )
    assert audits[0].frames_extended == 12


def test_offset_policy_none_leaves_voicing_untouched():
    n = 60
    arrays = _blank(n)
    frames = _note(arrays, 0, 0, 0, np.full(50, 110.0, dtype=np.float32))
    before = arrays[1].copy()

    f0, period = _f0(n, 0, frames, np.full(50, 110.0, dtype=np.float32), periodicity=0.0)

    _, voiced, audits = relabel_tails(
        *arrays, f0, period, stem="t", hop_seconds=HOP_SECONDS,
        offset_policy="none",
    )

    assert np.array_equal(voiced, before)
    assert audits[0].frames_trimmed == 0
    assert audits[0].frames_extended == 0


# ---------------------------------------------------------------------------
# Note boundaries
# ---------------------------------------------------------------------------

def test_adjacent_notes_are_not_merged():
    """Two notes touching with no unvoiced gap keep separate tails.

    A contiguous-run segmentation would treat these as one note and pollute
    the body reference with the other note's pitch — the same failure the
    guitarset overhang code uses note_ids to avoid.
    """
    n = 90
    arrays = _blank(n)
    first  = _note(arrays, 0, 0, 0,  np.full(40, 110.0, dtype=np.float32))
    second = _note(arrays, 0, 1, 40, np.full(40, 220.0, dtype=np.float32))

    f0 = np.zeros((N_STRINGS, n), dtype=np.float32)
    period = np.zeros((N_STRINGS, n), dtype=np.float32)
    f0[0, first]  = 55.0      # an octave error under note 0's body
    f0[0, second] = 110.0     # an octave error under note 1's body
    period[0, :80] = 0.9

    pitch, _, audits = relabel_tails(
        *arrays, f0, period, stem="t", hop_seconds=HOP_SECONDS,
        tail_policy="track", offset_policy="none",
    )

    assert len(audits) == 2
    assert audits[0].human_body_hz == pytest.approx(110.0)
    assert audits[1].human_body_hz == pytest.approx(220.0)
    # Each tail folds toward its own note's octave, not the other's.
    assert np.allclose(pitch[0, first[-8:]], 110.0, rtol=1e-3)
    assert np.allclose(pitch[0, second[-8:]], 220.0, rtol=1e-3)


def test_short_note_has_no_tail_but_still_gets_an_offset():
    """A note shorter than `divider` frames keeps its pitch and can still trim."""
    n = 30
    arrays = _blank(n)
    frames = _note(arrays, 0, 0, 0, np.full(4, 110.0, dtype=np.float32))
    original = arrays[0][0, frames].copy()

    f0, period = _f0(n, 0, frames, np.full(4, 220.0, dtype=np.float32))

    pitch, _, audits = relabel_tails(
        *arrays, f0, period, stem="t", hop_seconds=HOP_SECONDS,
        divider=5, offset_policy="none",
    )

    assert np.array_equal(pitch[0, frames], original)
    assert audits[0].n_frames == 4


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        (55.0, 110.0),    # an octave below folds up
        (220.0, 110.0),   # an octave above folds down
        (27.5, 110.0),    # two octaves below
        (116.5, 116.5),   # a semitone sharp is real, and left alone
        (110.0, 110.0),
    ],
)
def test_octave_fold(value, expected):
    folded, _ = octave_fold(np.array([value]), 110.0)
    assert folded[0] == pytest.approx(expected, rel=1e-6)


def test_summarise_reports_body_disagreement():
    n = 60
    arrays = _blank(n)
    frames = _note(arrays, 0, 0, 0, np.full(50, 110.0, dtype=np.float32))
    # An octave apart on the body: the disagreement worth surfacing.
    f0, period = _f0(n, 0, frames, np.full(50, 220.0, dtype=np.float32))

    _, _, audits = relabel_tails(
        *arrays, f0, period, stem="t", hop_seconds=HOP_SECONDS,
        offset_policy="none",
    )
    summary = summarise(audits)

    assert summary["notes"] == 1
    assert summary["body_disagreement_over_600c"] == 1.0
    assert summary["body_disagreement_cents"]["median"] == pytest.approx(1200.0, abs=1)


def test_invalid_policies_are_rejected():
    arrays = _blank(10)
    f0 = np.zeros((N_STRINGS, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="tail_policy"):
        relabel_tails(*arrays, f0, f0, stem="t", hop_seconds=HOP_SECONDS,
                      tail_policy="nonsense")
    with pytest.raises(ValueError, match="offset_policy"):
        relabel_tails(*arrays, f0, f0, stem="t", hop_seconds=HOP_SECONDS,
                      offset_policy="nonsense")
