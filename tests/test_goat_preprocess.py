"""Tests for GOAT preprocessing: transposition reconciliation and path resolution."""
import pytest

from resono.data.datasets.gaps.score import ScoreNote
from resono.data.datasets.goat.preprocess import _reconcile_transposition, _resolve


def score(*pitches: int) -> list[ScoreNote]:
    return [ScoreNote(1, 0, 3840, pitch, 1, 0) for pitch in pitches]


def midi(*pitches: int) -> list[tuple[float, float, int]]:
    return [(float(i), i + 0.5, pitch) for i, pitch in enumerate(pitches)]


# ---------------------------------------------------------------------------
# Transposition reconciliation
# ---------------------------------------------------------------------------

def test_agreeing_pitches_are_left_alone():
    notes = (40, 45, 50, 55, 59, 64)
    assert _reconcile_transposition(score(*notes), midi(*notes)) == 0


def test_octave_low_tablature_is_raised():
    # item_54 and item_95 are written an octave below their MIDI, and nothing
    # in the .gp5 records it: same instrument, tuning and capo as takes that
    # agree. Coverage on those two runs 11.9% and 22.9% without this.
    notes = (40, 45, 50, 55, 59, 64)
    shifted = tuple(pitch - 12 for pitch in notes)
    assert _reconcile_transposition(score(*shifted), midi(*notes)) == 12


def test_a_shift_beyond_an_octave_is_not_searched():
    # The pitches have distinct pairwise intervals, so the only shift that
    # can score at all is the true one. Ordinary music is nothing like this:
    # a scale is self-similar enough that several wrong shifts land a third
    # of the notes, which is why the search stops at an octave and demands a
    # margin rather than taking whichever shift scores best.
    distinct = (40, 41, 43, 47, 56, 69)
    assert _reconcile_transposition(
        score(*(pitch - 12 for pitch in distinct)), midi(*distinct)) == 12
    assert _reconcile_transposition(
        score(*(pitch - 13 for pitch in distinct)), midi(*distinct)) == 0


def test_an_unconvincing_shift_is_refused():
    # Half the take matches where it stands and the rest matches nothing.
    # Ordinary alignment failure must not be mistaken for a transposition:
    # the aligner can report a bad take, a silent pitch shift cannot.
    assert _reconcile_transposition(score(40, 43, 47, 48, 55, 62, 63, 70),
                                    midi(40, 43, 47, 48, 90, 91, 92, 93)) == 0


def test_repeated_pitches_are_weighted_by_how_often_they_are_played():
    # Counters, not sets: one open low E against forty of them should not
    # count for as much as the forty. A set-based overlap would score the
    # -12 shift 1/2 against +0's 1/2 and the margin would decide it by
    # accident; by count the +0 reading is overwhelming.
    tablature = score(*([64] * 40 + [52]))
    recording = midi(*([64] * 40 + [40]))
    assert _reconcile_transposition(tablature, recording) == 0


def test_an_empty_score_does_not_divide_by_zero():
    assert _reconcile_transposition([], midi(40, 45)) == 0


# ---------------------------------------------------------------------------
# metadata.csv path resolution
# ---------------------------------------------------------------------------

@pytest.fixture
def goat_root(tmp_path):
    take = tmp_path / "data" / "item_0"
    take.mkdir(parents=True)
    (take / "item_0.gp5").touch()
    return tmp_path


def test_the_leading_goat_directory_is_stripped(goat_root):
    # metadata.csv writes paths as GOAT/item_0/item_0.gp5, relative to the
    # archive root rather than to the directory the files actually sit in.
    resolved = _resolve(goat_root, "GOAT/item_0/item_0.gp5")
    assert resolved == goat_root / "data" / "item_0" / "item_0.gp5"


def test_a_blank_field_resolves_to_nothing(goat_root):
    # The 20 takes without fine-aligned MIDI leave that column empty.
    assert _resolve(goat_root, "") is None
    assert _resolve(goat_root, "   ") is None


def test_a_missing_file_resolves_to_nothing(goat_root):
    assert _resolve(goat_root, "GOAT/item_1/item_1.gp5") is None
