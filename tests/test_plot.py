"""Tests for the label-inspection plot: cache reading and lane labelling."""
import numpy as np
import pytest

from resono.data.plot import _lane_label, load_track, note_name


@pytest.fixture
def cache(tmp_path):
    """A two-frame, six-string cache with no onset file (GuitarSet-shaped)."""
    root = tmp_path / "somedataset"
    root.mkdir()
    np.save(root / "t-audio.npy", np.zeros(512, dtype=np.float32))
    np.save(root / "t-pitch.npy", np.zeros((6, 2), dtype=np.float32))
    np.save(root / "t-voiced.npy", np.zeros((6, 2), dtype=bool))
    return tmp_path


def test_note_name_round_trips_known_pitches():
    assert note_name(64) == "E4"      # high E
    assert note_name(40) == "E2"      # low E
    assert note_name(69) == "A4"      # concert A


def test_load_track_returns_none_onset_when_absent(cache):
    # GuitarSet writes no onset file; the plot degrades rather than failing.
    assert load_track(cache, "somedataset", "t")["onset"] is None


def test_load_track_reads_onset_when_present(cache):
    np.save(cache / "somedataset" / "t-onset.npy", np.zeros((6, 2), dtype=bool))
    assert load_track(cache, "somedataset", "t")["onset"] is not None


def test_load_track_names_what_is_missing(cache):
    (cache / "somedataset" / "t-voiced.npy").unlink()
    with pytest.raises(FileNotFoundError, match="voiced"):
        load_track(cache, "somedataset", "t")


def test_lane_label_shows_a_range_not_a_single_pitch():
    # A lone note would read as the open-string tuning, which it usually is
    # not — a range cannot be misread that way.
    pitch = np.array([[440.0, 493.88]], dtype=np.float32)
    voiced = np.array([[True, True]])
    assert _lane_label(pitch, voiced, 0) == "0   A4–B4"


def test_lane_label_collapses_a_constant_pitch():
    pitch = np.array([[440.0, 440.0]], dtype=np.float32)
    voiced = np.array([[True, True]])
    assert _lane_label(pitch, voiced, 0) == "0   A4"


def test_lane_label_marks_a_silent_string():
    pitch = np.zeros((1, 2), dtype=np.float32)
    voiced = np.zeros((1, 2), dtype=bool)
    assert _lane_label(pitch, voiced, 0) == "0   silent"


def test_lane_label_ignores_unvoiced_frames():
    # Unvoiced frames hold 0 Hz; letting them into the range would produce a
    # nonsensical bottom note.
    pitch = np.array([[0.0, 440.0]], dtype=np.float32)
    voiced = np.array([[False, True]])
    assert _lane_label(pitch, voiced, 0) == "0   A4"
