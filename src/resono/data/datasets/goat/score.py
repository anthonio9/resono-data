"""Read GOAT's Guitar Pro tablature into the ScoreNote form the aligner expects.

GOAT ships the tablature three ways: .gp5, .gp, and a DadaGP token text. The
.gp5 is the source — the paper provides it "to allow for conversion into the
DadaGP format", and the MIDI comes from Guitar Pro's own export of the same
file. Reading it means matching the MIDI against its ancestor rather than
against a sibling derivation, which is worth a great deal: excluding tie
notes, the .gp5 note count matches the MIDI exactly on 132 of 172 takes and
to within 1% on 153, where the DadaGP text managed 36 of 166.

Ties are the discrepancy. A tied note continues the previous one and Guitar
Pro's MIDI export merges the pair, so a tie is not a new note: item_0 holds
1418 normal + 97 tie + 34 dead notes, and 1549 - 97 is exactly the MIDI's
1452. Dead (muted) notes do sound and are kept.

Tuning is read per string rather than assumed. Six distinct tunings appear
across the corpus, and metadata.csv is not a reliable guide: item_0 is
labelled "standard" while its .gp5 tunes the sixth string to D.

String 1 is the *high* E, matching MusicXML rather than resono, whose index 0
is the low E. This module deliberately keeps Guitar Pro's numbering so the
GAPS aligner and assign_strings can be reused unchanged.

The written score is not the performance. Guitar Pro stores a repeated
section once and a tempo change as an event, while the MIDI export plays the
repeat out and follows the tempo; reading the file literally puts the two out
of step. Six takes are affected — one repeat and five tempo changes — and
between them they account for every timing failure in the corpus.
"""
from dataclasses import dataclass
from pathlib import Path

import guitarpro

from resono.data.datasets.gaps.score import ScoreNote

#: Guitar Pro counts ticks in these units.
QUARTER_TICKS = guitarpro.Duration.quarterTime

#: A tie continues the previous note; the MIDI export merges them.
SILENT_NOTE_TYPES = {"tie"}


@dataclass(frozen=True)
class Tablature:
    """One take's tablature: notes, their played ticks, tempo map and tuning."""
    notes: list[ScoreNote]
    ticks: list[int]            # parallel to notes, in playing order from 0
    tempos: list[tuple[int, int]]   # (tick, bpm); the first entry is the song tempo
    tuning: dict[int, int]      # Guitar Pro string number -> open MIDI pitch


def unfold_repeats(measures: list) -> list[int]:
    """Measure indices in playing order, with repeated sections written out.

    Guitar Pro stores a repeated section once; its MIDI export plays it out,
    so folded measures leave the tablature short. One GOAT take repeats
    (item_120, measures 33-36), and reading it folded costs 676 of its 3254
    notes and 23 seconds of its length.

    This is the GP counterpart of gaps.score.unfold_repeats and returns the
    same thing, but reads a different encoding: MusicXML marks repeats with
    barline elements, GP with `isRepeatOpen` and `repeatClose` on the measure
    header. `repeatClose` counts *extra* passes, so item_120's 2 means the
    section is played three times — which is what its MIDI length confirms.

    Alternate endings occur nowhere in GOAT, so rather than guess at how they
    interact with the pass count this refuses the take and lets preprocessing
    skip it with a reason in the manifest.
    """
    order: list[int] = []
    taken: dict[int, int] = {}
    index = section = 0
    while index < len(measures):
        header = measures[index].header
        if header.repeatAlternative:
            raise ValueError(
                f"measure {header.number} has an alternate ending, which is "
                "not unfolded")
        if header.isRepeatOpen:
            section = index
        order.append(index)
        if header.repeatClose > 0 and taken.get(index, 0) < header.repeatClose:
            taken[index] = taken.get(index, 0) + 1
            index = section
            continue
        index += 1
    return order


def read_score(path: Path) -> Tablature:
    """Parse a Guitar Pro file into ScoreNotes plus the timing they need.

    Every GOAT take holds exactly one 6-string track — the authors split
    multi-guitar tabs into separate files — so the first track is taken and a
    second, if one ever appears, is ignored rather than merged, since only one
    part was recorded.
    """
    song = guitarpro.parse(str(path))
    if not song.tracks:
        raise ValueError(f"{path} has no tracks")
    track = song.tracks[0]
    # Track.offset is the capo fret: it raises every sounding pitch without
    # changing the written fret numbers. One GOAT take (item_98) uses one, and
    # ignoring it detunes that take by a whole tone.
    tuning = {string.number: string.value + track.offset
              for string in track.strings}

    notes: list[ScoreNote] = []
    ticks: list[int] = []
    tempos: list[tuple[int, int]] = [(0, song.tempo)]

    # `cursor` is elapsed playing time, so a measure visited twice emits its
    # notes at two different ticks from the one stored copy. Beat positions
    # are taken relative to their measure, which also drops Guitar Pro's habit
    # of putting the first beat at one quarter rather than at zero.
    cursor = 0
    for index in unfold_repeats(track.measures):
        measure = track.measures[index]
        for voice in measure.voices:
            for beat in voice.beats:
                tick = cursor + (beat.start - measure.start)
                change = beat.effect.mixTableChange
                if change is not None and change.tempo is not None:
                    tempos.append((tick, change.tempo.value))
                for note in beat.notes:
                    if note.type.name in SILENT_NOTE_TYPES:
                        continue
                    if note.string not in tuning:
                        continue
                    notes.append(ScoreNote(
                        measure=measure.header.number,
                        offset=tick - cursor,
                        measure_len=max(1, measure.length),
                        pitch=tuning[note.string] + note.value,
                        string=note.string,
                        fret=note.value,
                    ))
                    ticks.append(tick)
        cursor += measure.length

    # Voices interleave, so sort into playing order and keep ticks parallel.
    order = sorted(range(len(notes)), key=lambda i: (ticks[i], notes[i].pitch))
    return Tablature(
        notes=[notes[i] for i in order],
        ticks=[ticks[i] for i in order],
        tempos=sorted(tempos),
        tuning=tuning,
    )


def predict_times(
    ticks: list[int], tempos: list[tuple[int, int]], ticks_per_beat: int = 960
) -> list[float]:
    """Seconds at which each ScoreNote is notated to sound.

    GOAT's performers played to the tablature's click — measured over the
    takes that aligned, the fine-aligned MIDI ends within 0.05% of the
    unaligned one — so the notated timeline needs no warping and the notated
    tempo predicts onsets well enough for the aligner's tolerance window.
    This stands in for the syncpoint interpolation GAPS needs and GOAT has no
    equivalent of.

    `tempos` is the map read from the file rather than a single value. Five
    takes change tempo mid-piece through mix-table events, and a constant
    tempo drifts from the first change onward until it leaves the tolerance
    window, at which point matching simply stops: item_115 loses half its
    notes that way and item_91 half of those.
    """
    # Seconds elapsed at the start of each tempo section, accumulated once.
    elapsed = [0.0]
    for (tick, bpm), (next_tick, _) in zip(tempos, tempos[1:]):
        elapsed.append(
            elapsed[-1] + (next_tick - tick) * 60.0 / (bpm * ticks_per_beat))

    times: list[float] = []
    section = 0
    for tick in ticks:                  # sorted, so the section only moves forward
        while section + 1 < len(tempos) and tempos[section + 1][0] <= tick:
            section += 1
        start, bpm = tempos[section]
        times.append(
            elapsed[section] + (tick - start) * 60.0 / (bpm * ticks_per_beat))
    return times
