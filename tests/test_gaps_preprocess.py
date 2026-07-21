"""Tests for GAPS preprocessing: measure cross-check and exclusion manifest."""
from resono.data.datasets.gaps.preprocess import _check_measures
from resono.data.datasets.gaps.score import ScoreNote


def notes(n_measures: int) -> list[ScoreNote]:
    return [ScoreNote(m, 0, 8, 60, 1, 0) for m in range(n_measures)]


def syncpoints(n: int) -> list[list]:
    return [[i, float(i)] for i in range(n)]


def test_agreeing_counts_pass():
    agree, performed, expected = _check_measures(notes(32), syncpoints(32))
    assert agree and performed == 32 and expected == 32


def test_off_by_one_is_tolerated():
    # Anacrusis handling routinely shifts the count by one; that alone is not
    # evidence the mapping is wrong.
    agree, _, _ = _check_measures(notes(235), syncpoints(236))
    assert agree


def test_larger_disagreement_is_flagged():
    agree, performed, expected = _check_measures(notes(88), syncpoints(152))
    assert not agree
    assert performed == 88 and expected == 152


def test_sub_measure_syncpoints_do_not_inflate_the_count():
    # 3-tuples are within-measure refinements, not extra measures.
    points = [[0, 0.0], [1, 1.0], [1, 1.5, 160], [2, 2.0]]
    agree, _, expected = _check_measures(notes(3), points)
    assert agree and expected == 3


def test_missing_syncpoints_do_not_disqualify():
    agree, performed, expected = _check_measures(notes(10), [])
    assert agree and performed == expected == 10


# ---------------------------------------------------------------------------
# Chord-onset grouping
# ---------------------------------------------------------------------------

from resono.data.datasets.gaps.align import _group_onsets


def test_rolled_chord_notes_share_a_group():
    # A rolled chord arrives over a few milliseconds; all of it must sort as
    # one onset so the members order by pitch rather than by roll direction.
    assert _group_onsets([1.00, 1.02, 1.04], 0.05) == [1.0, 1.0, 1.0]


def test_notes_beyond_the_window_start_a_new_group():
    assert _group_onsets([1.00, 1.02, 1.20], 0.05) == [1.0, 1.0, 1.2]


def test_groups_do_not_chain_indefinitely():
    # Each group is measured from its own start, so a steady stream of notes
    # 40 ms apart must not collapse into one group.
    assert _group_onsets([0.0, 0.04, 0.08, 0.12], 0.05) == [0.0, 0.0, 0.08, 0.08]


def test_zero_window_leaves_onsets_untouched():
    assert _group_onsets([1.0, 1.02], 0.0) == [1.0, 1.02]


def test_empty_input_is_handled():
    assert _group_onsets([], 0.05) == []
