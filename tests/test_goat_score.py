"""Tests for GOAT Guitar Pro reading: ties, capo, tuning, tick ordering.

The fixtures build .gp5 files rather than checking one in, so the conventions
under test are stated in the test itself. Guitar Pro stores durations, not
absolute positions: a beat's `start` is recomputed on read from the beats
before it in the same voice, so placing a note later in a measure means
writing the beats that precede it.
"""
import guitarpro
import pytest
from guitarpro import models as gp

from resono.data.datasets.goat.score import (
    predict_times, read_score, unfold_repeats)

QUARTER = guitarpro.Duration.quarterTime
STANDARD = (64, 59, 55, 50, 45, 40)          # string 1 (high E) to string 6


def _track(tempo, tuning, capo):
    song = gp.Song(tempo=tempo)
    song.tracks, song.measureHeaders = [], []
    track = gp.Track(song, 1, offset=capo, strings=[
        gp.GuitarString(number, value)
        for number, value in enumerate(tuning, start=1)])
    track.measures = []
    song.tracks.append(track)
    return song, track


def _measure(song, track, number, beats_per_measure=4):
    header = gp.MeasureHeader(
        number=number, start=QUARTER + (number - 1) * beats_per_measure * QUARTER)
    header.timeSignature.numerator = beats_per_measure
    song.measureHeaders.append(header)
    measure = gp.Measure(track, header)
    track.measures.append(measure)
    return measure


def _quarter(voice):
    beat = gp.Beat(voice, duration=gp.Duration(4))
    beat.status = gp.BeatStatus.normal
    voice.beats.append(beat)
    return beat


def write_score(
    path,
    voices=(((("normal", (1, 0)),),),),
    tuning=STANDARD,
    capo=0,
    tempo=120,
    beats_per_measure=4,
):
    """Write a one-measure .gp5.

    `voices` is one sequence of beats per voice; each beat is a sequence of
    (note type, (string, fret)) pairs, and an empty beat is a rest that still
    advances the clock. Every beat is a quarter note.
    """
    song, track = _track(tempo, tuning, capo)
    measure = _measure(song, track, 1, beats_per_measure)

    for index, beats in enumerate(voices):
        voice = measure.voices[index]
        for beat_spec in beats:
            beat = _quarter(voice)
            for kind, (string, fret) in beat_spec:
                beat.notes.append(gp.Note(
                    beat, value=fret, string=string,
                    type=getattr(gp.NoteType, kind)))

    guitarpro.write(song, str(path))
    return path


def write_structure(path, n_measures, repeat=None, tempo_changes=(), tempo=120):
    """Write an n-measure .gp5 carrying one quarter note per measure.

    Measure n holds fret n on the high E, so the frets read back name the
    measures in the order they were played. `repeat` is
    (open measure, close measure, extra passes) and `tempo_changes` is a
    sequence of (measure, bpm) applied at that measure's first beat, both
    numbered from 1.
    """
    song, track = _track(tempo, STANDARD, capo=0)
    changes = dict(tempo_changes)

    for number in range(1, n_measures + 1):
        measure = _measure(song, track, number)
        if repeat is not None:
            opens, closes, passes = repeat
            measure.header.isRepeatOpen = number == opens
            if number == closes:
                measure.header.repeatClose = passes
        beat = _quarter(measure.voices[0])
        if number in changes:
            change = gp.MixTableChange()
            change.tempo = gp.MixTableItem(
                value=changes[number], duration=0, allTracks=False)
            beat.effect.mixTableChange = change
        beat.notes.append(gp.Note(
            beat, value=number, string=1, type=gp.NoteType.normal))

    guitarpro.write(song, str(path))
    return path


@pytest.fixture
def score(tmp_path):
    def build(**kwargs):
        return read_score(write_score(tmp_path / "take.gp5", **kwargs))
    return build


@pytest.fixture
def structure(tmp_path):
    def build(*args, **kwargs):
        return read_score(write_structure(tmp_path / "take.gp5", *args, **kwargs))
    return build


# ---------------------------------------------------------------------------
# Which notes sound
# ---------------------------------------------------------------------------

def test_tie_note_is_not_a_new_note(score):
    # A tie continues the previous note and Guitar Pro's MIDI export merges
    # the pair, so counting it would put the tablature ahead of the MIDI.
    # Across GOAT this is the whole discrepancy: item_0 holds 1418 normal + 97
    # tie + 34 dead notes and its MIDI holds 1549 - 97 = 1452.
    tablature = score(voices=((
        (("normal", (1, 0)),),
        (("tie", (1, 0)),),
        (("normal", (1, 3)),),
    ),))
    assert [note.fret for note in tablature.notes] == [0, 3]


def test_dead_notes_are_kept(score):
    # A dead (muted) note is percussive but it does sound, and Guitar Pro
    # exports it, so dropping it would put the tablature behind the MIDI.
    tablature = score(voices=(((("dead", (6, 5)),),),))
    assert [note.fret for note in tablature.notes] == [5]


def test_rest_advances_the_clock_without_a_note(score):
    tablature = score(voices=((
        (("normal", (1, 0)),),
        (),
        (("normal", (1, 3)),),
    ),))
    assert len(tablature.notes) == 2
    assert tablature.ticks == [0, 2 * QUARTER]


# ---------------------------------------------------------------------------
# Pitch, tuning and capo
# ---------------------------------------------------------------------------

def test_string_numbering_follows_guitar_pro(score):
    # String 1 is the high E, as in MusicXML and unlike resono, whose index 0
    # is the low E. This module keeps Guitar Pro's numbering so the GAPS
    # aligner and assign_strings can be reused unchanged.
    tablature = score()
    assert tablature.tuning[1] == 64
    assert tablature.tuning[6] == 40


def test_pitch_is_tuning_plus_fret(score):
    tablature = score(voices=(((("normal", (6, 3)),),),))
    assert tablature.notes[0].pitch == 43       # low E + 3 semitones


def test_tuning_is_read_per_string_not_assumed(score):
    # metadata.csv is not a reliable guide: item_0 is labelled "standard"
    # while its .gp5 tunes the sixth string to D.
    dropped_d = (64, 59, 55, 50, 45, 38)
    tablature = score(tuning=dropped_d, voices=(((("normal", (6, 0)),),),))
    assert tablature.tuning[6] == 38
    assert tablature.notes[0].pitch == 38


def test_capo_raises_every_pitch(score):
    # Track.offset is the capo fret: it raises what sounds without changing
    # the written fret numbers. item_98 uses one, and ignoring it detunes the
    # whole take by a whole tone.
    tablature = score(capo=2, voices=(((("normal", (1, 0)), ("normal", (6, 5))),),))
    assert tablature.tuning == {1: 66, 2: 61, 3: 57, 4: 52, 5: 47, 6: 42}
    assert sorted(note.pitch for note in tablature.notes) == [47, 66]
    assert sorted(note.fret for note in tablature.notes) == [0, 5]


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def test_ticks_start_at_zero(score):
    # Guitar Pro puts the first beat at one quarter, not at zero.
    tablature = score(voices=(((("normal", (1, 0)),), (("normal", (1, 3)),)),))
    assert tablature.ticks == [0, QUARTER]


def test_simultaneous_notes_sort_by_pitch(score):
    # Voices interleave, and the aligner's monotonicity depends on both
    # sequences ordering a chord the same way.
    tablature = score(voices=(
        ((("normal", (1, 0)),),),          # high E, MIDI 64
        ((("normal", (6, 0)),),),          # low E,  MIDI 40
    ))
    assert tablature.ticks == [0, 0]
    assert [note.pitch for note in tablature.notes] == [40, 64]


def test_measure_length_uses_the_time_signature(score):
    tablature = score(beats_per_measure=3)
    assert all(note.measure_len == 3 * QUARTER for note in tablature.notes)


def test_offset_is_measured_from_the_measure_start(score):
    tablature = score(voices=(((("normal", (1, 0)),), (("normal", (1, 3)),)),))
    assert [note.offset for note in tablature.notes] == [0, QUARTER]


def test_predict_times_scales_ticks_by_tempo():
    # One quarter at 120 bpm is half a second.
    assert predict_times([0, QUARTER, 2 * QUARTER], [(0, 120)]) == [0.0, 0.5, 1.0]


def test_predict_times_is_slower_at_a_lower_tempo():
    assert predict_times([QUARTER], [(0, 60)]) == [1.0]


def test_predict_times_follows_a_tempo_change():
    # Ticks before the change keep the old rate; ticks after it are measured
    # from where the change fell, not from zero.
    tempos = [(0, 120), (2 * QUARTER, 240)]
    ticks = [0, QUARTER, 2 * QUARTER, 3 * QUARTER, 4 * QUARTER]
    assert predict_times(ticks, tempos) == [0.0, 0.5, 1.0, 1.25, 1.5]


def test_predict_times_handles_consecutive_changes():
    tempos = [(0, 120), (QUARTER, 240), (2 * QUARTER, 60)]
    assert predict_times([0, QUARTER, 2 * QUARTER, 3 * QUARTER], tempos) == [
        0.0, 0.5, 0.75, 1.75]


# ---------------------------------------------------------------------------
# Tempo map
# ---------------------------------------------------------------------------

def test_the_tempo_map_starts_with_the_song_tempo(structure):
    tablature = structure(2, tempo=90)
    assert tablature.tempos == [(0, 90)]


def test_a_mid_piece_tempo_change_is_read(structure):
    # Five GOAT takes change tempo through mix-table events. Reading only the
    # song tempo drifts from the change onward until the prediction leaves
    # the aligner's tolerance, at which point matching stops: item_115 ends up
    # 57 seconds long against its MIDI's 110 and loses half its notes.
    tablature = structure(3, tempo=120, tempo_changes=((2, 240),))
    assert tablature.tempos == [(0, 120), (4 * QUARTER, 240)]

    times = predict_times(tablature.ticks, tablature.tempos)
    assert times == [0.0, 2.0, 3.0]     # 4 quarters at 120, then 4 at 240


# ---------------------------------------------------------------------------
# Repeats
# ---------------------------------------------------------------------------

class FakeHeader:
    def __init__(self, number, opens=False, closes=-1, alternative=0):
        self.number = number
        self.isRepeatOpen = opens
        self.repeatClose = closes
        self.repeatAlternative = alternative


class FakeMeasure:
    def __init__(self, header):
        self.header = header


def measures(*headers) -> list[FakeMeasure]:
    return [FakeMeasure(header) for header in headers]


def test_no_repeats_is_document_order():
    assert unfold_repeats(measures(
        FakeHeader(1), FakeHeader(2), FakeHeader(3))) == [0, 1, 2]


def test_repeat_close_counts_extra_passes():
    # Guitar Pro's repeatClose is the number of times the section is played
    # *again*, so 2 means three plays. item_120's repeat spans four measures
    # and adds 23.1 seconds to a 156-second piece, which is two extra passes.
    assert unfold_repeats(measures(
        FakeHeader(1),
        FakeHeader(2, opens=True),
        FakeHeader(3, closes=2),
        FakeHeader(4),
    )) == [0, 1, 2, 1, 2, 1, 2, 3]


def test_a_single_repeat_plays_the_section_twice():
    assert unfold_repeats(measures(
        FakeHeader(1, opens=True), FakeHeader(2, closes=1))) == [0, 1, 0, 1]


def test_a_close_without_an_open_repeats_from_the_start():
    assert unfold_repeats(measures(
        FakeHeader(1), FakeHeader(2, closes=1), FakeHeader(3))) == [0, 1, 0, 1, 2]


def test_consecutive_sections_each_repeat_on_their_own():
    assert unfold_repeats(measures(
        FakeHeader(1, opens=True), FakeHeader(2, closes=1),
        FakeHeader(3, opens=True), FakeHeader(4, closes=1),
    )) == [0, 1, 0, 1, 2, 3, 2, 3]


def test_an_alternate_ending_is_refused():
    # None occurs in GOAT, so rather than guess how one interacts with the
    # pass count this raises and lets preprocessing skip the take with a
    # reason in the manifest.
    with pytest.raises(ValueError, match="alternate ending"):
        unfold_repeats(measures(
            FakeHeader(1, opens=True), FakeHeader(2, closes=1, alternative=1)))


def test_repeated_measures_sound_again_at_later_ticks(structure):
    # The repeat is stored once and played three times, so the notes are
    # emitted three times from the one stored copy — which is what makes the
    # tablature note count match the MIDI's.
    tablature = structure(4, repeat=(2, 3, 2))
    assert [note.fret for note in tablature.notes] == [1, 2, 3, 2, 3, 2, 3, 4]
    assert tablature.ticks == [i * 4 * QUARTER for i in range(8)]


def test_a_repeated_measure_keeps_its_offset_within_the_measure(structure):
    tablature = structure(3, repeat=(1, 2, 1))
    assert all(note.offset == 0 for note in tablature.notes)
