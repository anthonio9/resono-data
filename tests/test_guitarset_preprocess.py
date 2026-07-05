import jams
import numpy as np
import pytest

from resono.data.datasets.guitarset.preprocess import (
    extract_pitch_array_jams,
    extract_pitch_note_arrays_jams,
    remove_pitch_overhangs,
)

HOP_SECONDS = 256 / 44100


# ---------------------------------------------------------------------------
# JAMS fixture helpers
# ---------------------------------------------------------------------------

def _make_single_note_jams(body_freq, tail_freqs, hop_seconds=HOP_SECONDS):
    """One string, one note: 20 body frames at body_freq, then tail_freqs."""
    jam = jams.JAMS()
    jam.file_metadata.duration = (len(tail_freqs) + 20) * hop_seconds

    ann = jams.Annotation(namespace="pitch_contour")
    t = 0.0
    for _ in range(20):
        ann.append(time=t, duration=hop_seconds,
                   value={"frequency": body_freq, "voiced": True, "index": 0},
                   confidence=1.0)
        t += hop_seconds
    for freq in tail_freqs:
        ann.append(time=t, duration=hop_seconds,
                   value={"frequency": freq, "voiced": True, "index": 0},
                   confidence=1.0)
        t += hop_seconds
    jam.annotations.append(ann)
    return jam


def _make_two_note_jams(freq_a, tail_a, freq_b, n_b=20, hop_seconds=HOP_SECONDS):
    """One string, two *adjacent* notes (no unvoiced gap between them).

    Note 0 (index 0): 20 body frames at freq_a, then tail_a drifting frames.
    Note 1 (index 1): n_b frames at freq_b, starting the very next frame.

    This is the case that a contiguous-run segmentation would wrongly merge.
    """
    jam = jams.JAMS()
    total = 20 + len(tail_a) + n_b
    jam.file_metadata.duration = total * hop_seconds

    ann = jams.Annotation(namespace="pitch_contour")
    t = 0.0
    for _ in range(20):
        ann.append(time=t, duration=hop_seconds,
                   value={"frequency": freq_a, "voiced": True, "index": 0},
                   confidence=1.0)
        t += hop_seconds
    for freq in tail_a:
        ann.append(time=t, duration=hop_seconds,
                   value={"frequency": freq, "voiced": True, "index": 0},
                   confidence=1.0)
        t += hop_seconds
    for _ in range(n_b):
        ann.append(time=t, duration=hop_seconds,
                   value={"frequency": freq_b, "voiced": True, "index": 1},
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
# Uniform-grid interpolation
# ---------------------------------------------------------------------------

def test_uniform_grid_fills_gaps_at_finer_hop():
    """A sustained note must have no interior unvoiced holes when the target
    grid is finer than GuitarSet's native hop. Nearest-frame assignment would
    leave every other frame unvoiced; interpolation fills them."""
    jam   = _make_single_note_jams(220.0, tail_freqs=[])   # 20 native-hop frames
    finer = HOP_SECONDS / 2
    n_frames = int(float(jam.file_metadata.duration) / finer)

    _, voiced = extract_pitch_array_jams(jam, finer, n_frames)
    v = voiced[0]

    first, last = np.where(v)[0][[0, -1]]
    assert v[first:last + 1].all(), "Interpolation left holes inside a sustained note"
    # Finer grid should roughly double the number of voiced frames vs native.
    assert (last - first) > 30, "Finer grid did not densify the note"


def test_note_ids_tag_distinct_notes():
    """Two adjacent notes must receive different per-frame note ids."""
    jam = _make_two_note_jams(220.0, tail_a=[], freq_b=330.0)
    n_frames = int(float(jam.file_metadata.duration) / HOP_SECONDS)

    _, voiced, note_ids = extract_pitch_note_arrays_jams(jam, HOP_SECONDS, n_frames)
    ids = np.unique(note_ids[0][note_ids[0] >= 0])
    assert len(ids) == 2, "Adjacent notes were not tagged as separate notes"


# ---------------------------------------------------------------------------
# remove_pitch_overhangs
# ---------------------------------------------------------------------------

def test_overhang_drifting_tail_silenced():
    """Tail frames that deviate > threshold must be marked unvoiced."""
    body_freq  = 220.0
    drift_freq = body_freq * (2 ** (120 / 1200))   # ~120 cents, over threshold
    tail = [drift_freq] * 4                         # last 20% of a 24-frame note

    jam = _make_single_note_jams(body_freq, tail)
    n_frames = int(float(jam.file_metadata.duration) / HOP_SECONDS)

    pitch, voiced, note_ids = extract_pitch_note_arrays_jams(jam, HOP_SECONDS, n_frames)
    _, voiced = remove_pitch_overhangs(pitch, voiced, note_ids,
                                       divider=5, threshold_cents=15.0)

    assert not voiced[0, -4:].any(), "Drifting tail frames should be silenced"


def test_overhang_stable_tail_kept():
    """Tail frames within threshold must remain voiced."""
    body_freq   = 220.0
    stable_freq = body_freq * (2 ** (5 / 1200))    # 5 cents, within threshold
    tail = [stable_freq] * 4

    jam = _make_single_note_jams(body_freq, tail)
    n_frames = int(float(jam.file_metadata.duration) / HOP_SECONDS)

    pitch, voiced, note_ids = extract_pitch_note_arrays_jams(jam, HOP_SECONDS, n_frames)
    _, voiced = remove_pitch_overhangs(pitch, voiced, note_ids,
                                       divider=5, threshold_cents=15.0)

    assert voiced[0, -4:].any(), "Stable tail frames should remain voiced"


def test_overhang_does_not_modify_body():
    """The note body (non-tail frames) must never be silenced."""
    body_freq  = 220.0
    drift_freq = body_freq * (2 ** (200 / 1200))
    tail = [drift_freq] * 4

    jam = _make_single_note_jams(body_freq, tail)
    n_frames = int(float(jam.file_metadata.duration) / HOP_SECONDS)

    pitch, voiced, note_ids = extract_pitch_note_arrays_jams(jam, HOP_SECONDS, n_frames)
    body_before = voiced[0, :20].copy()

    _, voiced = remove_pitch_overhangs(pitch, voiced, note_ids,
                                       divider=5, threshold_cents=15.0)

    assert (voiced[0, :20] == body_before).all(), "Body frames were modified"


def test_overhang_honours_note_boundaries():
    """The whole point of using note ids: an overhang inside note 0 must be
    caught, and note 1's legitimate tail must NOT be silenced by a body-mean
    polluted from merging the two adjacent notes."""
    freq_a = 220.0
    freq_b = 330.0   # a perfect fifth up — 702 cents away, would pollute a mean
    drift  = freq_a * (2 ** (120 / 1200))
    tail_a = [drift] * 4

    jam = _make_two_note_jams(freq_a, tail_a, freq_b, n_b=20)
    n_frames = int(float(jam.file_metadata.duration) / HOP_SECONDS)

    pitch, voiced, note_ids = extract_pitch_note_arrays_jams(jam, HOP_SECONDS, n_frames)

    # note 0 occupies frames 0..23 (tail 20..23), note 1 occupies 24..43.
    note0 = np.where(note_ids[0] == 0)[0]
    note1 = np.where(note_ids[0] == 1)[0]

    _, voiced = remove_pitch_overhangs(pitch, voiced, note_ids,
                                       divider=5, threshold_cents=15.0)

    # Note 0's drifting tail (its last 4 frames) is silenced...
    assert not voiced[0, note0[-4:]].any(), "Note 0 overhang not caught"
    # ...while all of note 1 (stable) survives untouched.
    assert voiced[0, note1].all(), "Note 1 was wrongly silenced by merged body mean"
