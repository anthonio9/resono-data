"""MIDI note reading for GAPS.

GAPS ships ordinary Standard MIDI Files, so this is a thin adapter over mido
rather than a parser. Iterating a ``MidiFile`` yields messages already merged
across tracks with delta times converted to seconds, which accumulates
directly into the performance timeline the labels need — and unlike a
hand-rolled tick conversion it stays correct if a file carries tempo changes.
"""
from pathlib import Path

import mido


def read_notes(path: Path) -> list[tuple[float, float, int]]:
    """Read a MIDI file into (onset_seconds, offset_seconds, pitch) tuples.

    Overlapping notes of the same pitch are paired first-on to first-off,
    which is how the GAPS alignment emitted them. Notes left sounding at the
    end of the file are dropped, having no offset to pair with.

    Raises
    ------
    ValueError
        If the file cannot be read as a MIDI file. Preprocessing treats this
        as a skippable track rather than letting one bad file end the run.
    """
    try:
        midi_file = mido.MidiFile(Path(path))
        messages = list(midi_file)
    except Exception as error:  # mido raises OSError, EOFError, KeyError, ...
        raise ValueError(f"could not read {Path(path).name}: {error}") from error

    time = 0.0
    sounding: dict[int, list[float]] = {}
    notes: list[tuple[float, float, int]] = []

    for message in messages:
        time += message.time
        if message.type == "note_on" and message.velocity > 0:
            sounding.setdefault(message.note, []).append(time)
        elif message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        ):
            # A note-on with velocity 0 is the conventional note-off encoding.
            starts = sounding.get(message.note)
            if starts:
                notes.append((starts.pop(0), time, message.note))

    return sorted(notes)
