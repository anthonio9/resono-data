"""Tests for GAPS MusicXML reading: repeat unfolding, tuning, tablature."""
import xml.etree.ElementTree as ET

import pytest

from resono.data.datasets.gaps.score import (
    read_score,
    read_tuning,
    string_to_index,
    unfold_repeats,
)


def _measures(xml: str) -> list[ET.Element]:
    return ET.fromstring(f"<part>{xml}</part>").findall("measure")


FORWARD = '<barline location="left"><repeat direction="forward"/></barline>'
BACKWARD = '<barline location="right"><repeat direction="backward"/></barline>'


def volta(number: int, kind: str) -> str:
    location = "left" if kind == "start" else "right"
    return (
        f'<barline location="{location}">'
        f'<ending number="{number}" type="{kind}"/></barline>'
    )


def test_no_repeats_is_document_order():
    measures = _measures("<measure/><measure/><measure/>")
    assert unfold_repeats(measures) == [0, 1, 2]


def test_simple_repeat_is_played_twice():
    measures = _measures(
        f"<measure>{FORWARD}</measure><measure>{BACKWARD}</measure><measure/>"
    )
    assert unfold_repeats(measures) == [0, 1, 0, 1, 2]


def test_repeat_without_forward_barline_returns_to_the_start():
    # MusicXML treats a lone backward repeat as repeating from the beginning.
    measures = _measures(f"<measure/><measure>{BACKWARD}</measure>")
    assert unfold_repeats(measures) == [0, 1, 0, 1]


def test_voltas_take_first_ending_then_second():
    # Deriving the pass number from the nearest backward repeat drops the
    # second-time volta entirely, because it carries no repeat of its own.
    measures = _measures(
        f"<measure>{FORWARD}</measure>"
        f"<measure>{volta(1, 'start')}{volta(1, 'stop')}{BACKWARD}</measure>"
        f"<measure>{volta(2, 'start')}{volta(2, 'stop')}</measure>"
    )
    assert unfold_repeats(measures) == [0, 1, 0, 2]


def test_repeat_times_attribute_is_honoured():
    measures = _measures(
        f"<measure>{FORWARD}</measure>"
        '<measure><barline location="right">'
        '<repeat direction="backward" times="3"/></barline></measure>'
    )
    assert unfold_repeats(measures) == [0, 1, 0, 1, 0, 1]


def test_consecutive_sections_each_repeat_once():
    measures = _measures(
        f"<measure>{FORWARD}</measure><measure>{BACKWARD}</measure>"
        f"<measure>{FORWARD}</measure><measure>{BACKWARD}</measure>"
    )
    assert unfold_repeats(measures) == [0, 1, 0, 1, 2, 3, 2, 3]


def test_measure_carrying_both_repeat_marks_repeats_itself():
    # Both marks on one measure make it its own section: played, repeated
    # once, then the piece continues. The point is that it terminates.
    measures = _measures(
        f"<measure>{BACKWARD}{FORWARD}</measure>" + "<measure/>" * 3
    )
    assert unfold_repeats(measures) == [0, 0, 1, 2, 3]


# ---------------------------------------------------------------------------
# String numbering and tuning
# ---------------------------------------------------------------------------

def test_string_numbering_is_inverted():
    # MusicXML string 1 is the high E; resono index 0 is the low E.
    assert string_to_index(1) == 5
    assert string_to_index(6) == 0


SCORE = """<score-partwise>
  <part id="P1"><measure number="1">
    <attributes><divisions>2</divisions>
      <time><beats>4</beats><beat-type>4</beat-type></time>
    </attributes>
    <note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration></note>
  </measure></part>
  <part id="P2"><measure number="1">
    <attributes><divisions>2</divisions>
      <time><beats>4</beats><beat-type>4</beat-type></time>
      <staff-details>
        <staff-tuning line="1"><tuning-step>D</tuning-step><tuning-octave>2</tuning-octave></staff-tuning>
        <staff-tuning line="6"><tuning-step>E</tuning-step><tuning-octave>4</tuning-octave></staff-tuning>
      </staff-details>
    </attributes>
    <note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration>
      <notations><technical><string>1</string><fret>0</fret></technical></notations>
    </note>
    <note><rest/><duration>2</duration></note>
    <note><pitch><step>F</step><alter>1</alter><octave>2</octave></pitch><duration>4</duration>
      <notations><technical><string>6</string><fret>4</fret></technical></notations>
    </note>
  </measure></part>
</score-partwise>"""


@pytest.fixture
def score_file(tmp_path):
    path = tmp_path / "score.xml"
    path.write_text(SCORE)
    return path


def test_reads_only_the_tablature_part(score_file):
    notes = read_score(score_file)
    # The notation part duplicates the notes but carries no strings, so a
    # reader that took both parts would report every note twice.
    assert len(notes) == 2
    assert [note.string for note in notes] == [1, 6]


def test_pitch_comes_from_the_pitch_element_not_string_and_fret(score_file):
    notes = read_score(score_file)
    assert notes[0].pitch == 64          # E4
    assert notes[1].pitch == 42          # F#2, with alter applied


def test_rests_do_not_shift_following_onsets(score_file):
    notes = read_score(score_file)
    assert notes[0].offset == 0
    assert notes[1].offset == 4          # after a half-note note and a rest


def test_measure_length_uses_the_time_signature(score_file):
    notes = read_score(score_file)
    assert all(note.measure_len == 8 for note in notes)   # 4/4 at 2 divisions


def test_read_tuning_inverts_staff_line_numbering(score_file):
    tuning = read_tuning(score_file)
    # line 1 is the lowest string, which is <string> 6; this score is dropped-D.
    assert tuning[6] == 38               # D2
    assert tuning[1] == 64               # E4


def test_tuning_agrees_with_the_tablature(score_file):
    tuning = read_tuning(score_file)
    for note in read_score(score_file):
        assert tuning[note.string] + note.fret == note.pitch


# ---------------------------------------------------------------------------
# Malformed pitch
# ---------------------------------------------------------------------------

MALFORMED = """<score-partwise><part id="P1"><measure number="1">
  <attributes><divisions>1</divisions>
    <time><beats>4</beats><beat-type>4</beat-type></time></attributes>
  <note><pitch><step> </step><octave>4</octave></pitch><duration>1</duration>
    <notations><technical><string>1</string><fret>0</fret></technical></notations>
  </note>
  <note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration>
    <notations><technical><string>1</string><fret>0</fret></technical></notations>
  </note>
</measure></part></score-partwise>"""


def test_malformed_pitch_drops_the_note_not_the_track(tmp_path):
    # A few GAPS scores carry a <step> holding only whitespace. Losing the
    # whole recording over one bad note in several thousand is the wrong trade.
    path = tmp_path / "malformed.xml"
    path.write_text(MALFORMED)
    notes = read_score(path)
    assert [note.pitch for note in notes] == [64]
