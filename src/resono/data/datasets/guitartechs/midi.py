"""Read Guitar-TECHS MIDI into per-string note events.

The annotations come from a Fishman Triple Play pickup, which reports one MIDI
note per plucked string. Three properties of that export shape everything here,
and all three were established by measurement rather than from the paper:

**No pitch bend.** Not one ``pitchwheel`` or ``control_change`` message exists
in any file. A note therefore carries a single pitch for its whole duration —
fine for sustained notes, wrong during a bend, and slightly wrong during
vibrato (measured excursion ±12 cents).

**String lives in the track, not the channel.** Tracks are named
``e B G D A E``, high string first. Per-string channels survive in only some
files (Bendings and Vibrato use 0-5; Harmonics, PalmMute and PinchHarmonics
collapse to channel 0), so the track index is the reliable source — and it
runs opposite to resono's low-E-first axis, hence :data:`TRACK_TO_STRING`.

**Onsets arrive ~23 ms late.** The pickup has to hear enough of the string to
decide what note it is. Measured against audio attacks: -23.2, -22.3, -23.5 and
-25.6 ms across two players, two guitars and four takes, sd 5-8 ms. Left
uncorrected this teaches a systematic delay to an onset head, which is then
penalised at evaluation time against GuitarSet's hand-corrected onsets.
"""
from dataclasses import dataclass
from pathlib import Path

import mido

# Guitar-TECHS tracks run high-to-low (e B G D A E); resono indexes strings
# low-to-high with 0 = low E. Track 1 in the file is therefore string 5.
TRACK_NAMES     = ("e", "B", "G", "D", "A", "E")
TRACK_TO_STRING = (5, 4, 3, 2, 1, 0)
N_STRINGS       = 6

# Open-string MIDI numbers by resono string index, low E first. Used only to
# flag physically impossible notes, which do occur: three across the five P1
# technique files, isolated pickup tracking errors rather than a mapping fault.
OPEN_STRING_MIDI = (40, 45, 50, 55, 59, 64)

# Median measured latency of the Fishman note_on behind the audio attack.
ONSET_LATENCY_MS = 23.0


@dataclass(frozen=True)
class Note:
    """One MIDI note on one string.

    Not necessarily one *played* note: see ``continues``.
    """

    string: int      # resono index, 0 = low E
    midi: int
    start: float     # seconds, latency-corrected
    end: float
    continues: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


# A pitch that moves across a semitone boundary cannot be expressed in these
# files — there are no pitch-bend messages — so the pickup ends the note and
# starts a new one at the new pitch. A single plucked note therefore arrives as
# a staircase of note_on events: one bend in P1 produced nine, of which the
# audio shows only the first has an attack.
#
# Runs are joined when the gap between them is under this. Measured against
# the authors' own event counts (paper Table II), 50 ms reproduces them within
# five notes on seven of eleven takes, including P2 Bendings at 29 against a
# stated 30. It does not rescue every take: P1 Bendings still yields ~105
# events against 69 counted by ear, because there the pickup drops out for up
# to two seconds mid-bend while the string rings on, and no timing threshold
# separates that from a genuine rest. Such takes are better excluded than
# fitted to.
MERGE_GAP_MS = 50.0


def read_notes(
    path: Path,
    onset_latency_ms: float = ONSET_LATENCY_MS,
    merge_gap_ms: float = MERGE_GAP_MS,
) -> list[Note]:
    """Parse one Guitar-TECHS .mid into per-string notes, in time order.

    Parameters
    ----------
    onset_latency_ms:
        Shifted off every note's start and end, to undo the pickup's detection
        delay. The whole note moves, so durations are preserved: only the
        onset was measured, and a release has no comparably sharp audio event
        to measure against. Pass 0.0 to keep the file's own timing.
    merge_gap_ms:
        Notes following within this of the previous note's end, on the same
        string, are marked ``continues`` — the same played note whose pitch
        crossed a semitone boundary. See :data:`MERGE_GAP_MS`. Nothing is
        discarded: each note keeps its own pitch and span, so a bend stays a
        rising staircase; only the onset is suppressed. Pass 0.0 to treat
        every note_on as a fresh pluck.
    """
    midi = mido.MidiFile(str(path))
    seconds_at = _tempo_map(midi)
    shift = onset_latency_ms / 1000.0

    notes: list[Note] = []
    for track_index, track in enumerate(midi.tracks[1:]):
        if track_index >= N_STRINGS:
            break
        string = TRACK_TO_STRING[track_index]

        tick = 0
        open_notes: dict[int, int] = {}
        for message in track:
            tick += message.time
            if message.type == "note_on" and message.velocity > 0:
                open_notes[message.note] = tick
            elif message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            ):
                start_tick = open_notes.pop(message.note, None)
                if start_tick is None:
                    continue
                start = seconds_at(start_tick) - shift
                end = seconds_at(tick) - shift
                if end > max(start, 0.0):
                    notes.append(Note(string, message.note, max(start, 0.0), end))

    notes.sort(key=lambda n: (n.start, n.string))
    return _mark_continuations(notes, merge_gap_ms / 1000.0)


def _mark_continuations(notes: list[Note], gap: float) -> list[Note]:
    """Flag notes that continue the previous one rather than starting anew."""
    if gap <= 0:
        return notes

    last_end: dict[int, float] = {}
    out = []
    for note in notes:
        previous = last_end.get(note.string)
        continues = previous is not None and note.start - previous < gap
        out.append(
            Note(note.string, note.midi, note.start, note.end, continues)
            if continues else note
        )
        last_end[note.string] = note.end
    return out


def played_notes(notes: list[Note]) -> int:
    """How many actual plucks these MIDI notes represent."""
    return sum(1 for n in notes if not n.continues)


def _tempo_map(midi: mido.MidiFile):
    """Return a tick -> seconds function honouring every tempo change.

    Every file observed carries a single set_tempo of 1,000,000 us/beat
    (60 BPM) at tick 0, but reading the map rather than assuming it is what
    catches the mistake cheaply: assuming the MIDI default of 120 BPM halves
    every timestamp, which misaligns the labels against the audio without
    raising anything.
    """
    changes = []
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type == "set_tempo":
                changes.append((tick, message.tempo))
    changes.sort()
    if not changes or changes[0][0] != 0:
        changes.insert(0, (0, 500000))

    ticks_per_beat = midi.ticks_per_beat

    def seconds_at(tick: int) -> float:
        total = 0.0
        for i, (start_tick, tempo) in enumerate(changes):
            if start_tick >= tick:
                break
            end_tick = min(tick, changes[i + 1][0] if i + 1 < len(changes) else tick)
            total += mido.tick2second(end_tick - start_tick, ticks_per_beat, tempo)
        return total

    return seconds_at


def implausible_notes(notes: list[Note]) -> list[Note]:
    """Notes below their string's open pitch, which no fretting can produce."""
    return [n for n in notes if n.midi < OPEN_STRING_MIDI[n.string]]
