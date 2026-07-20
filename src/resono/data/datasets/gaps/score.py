"""MusicXML score reading for GAPS: tablature, repeat unfolding, tuning.

GAPS scores contain two parts with identical note content — a standard
notation part and a tablature part. Only the tablature part carries
``<string>``/``<fret>``, and it carries them for every sounding note, so all
string information is read from there and pitch is read from ``<pitch>``
rather than derived from string+fret.
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

# Semitones above C for each diatonic step.
_STEP_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# MusicXML numbers strings 1 = highest-pitched (high E) through 6 = low E.
# resono indexes strings 0 = low E through 5 = high E, so the two run in
# opposite directions. Getting this backwards is silent — the labels stay
# self-consistent and only string_assignment_accuracy degrades — so the
# conversion lives in exactly one place.
N_STRINGS = 6


def string_to_index(musicxml_string: int) -> int:
    """Convert a MusicXML string number (1 = high E) to a resono index (0 = low E)."""
    return N_STRINGS - musicxml_string


@dataclass(frozen=True)
class ScoreNote:
    """One sounding note of the unfolded tablature part."""
    measure: int        # performed measure index (repeats expanded)
    offset: int         # onset within the measure, in divisions
    measure_len: int    # length of that measure, in divisions
    pitch: int          # MIDI note number, read from <pitch>
    string: int         # MusicXML string number, 1 = high E
    fret: int

    @property
    def position(self) -> float:
        """Score position in fractional measures, for interpolating onto time."""
        return self.measure + self.offset / max(1, self.measure_len)


def read_score(path: Path) -> list[ScoreNote]:
    """Read the tablature part of a GAPS MusicXML file, with repeats unfolded.

    Notes without both a string and a fret are dropped: they carry no usable
    label. In GAPS this removes nothing but rests, since the tablature part
    annotates every sounding note.
    """
    root = ET.parse(path).getroot()
    part = _tablature_part(root)
    if part is None:
        raise ValueError(f"{path} has no part carrying <string> annotations")

    measures = part.findall("measure")
    attributes = _scan_attributes(measures)
    order = unfold_repeats(measures)

    notes: list[ScoreNote] = []
    for performed_index, source_index in enumerate(order):
        divisions, measure_len = attributes[source_index]
        for offset, pitch, string, fret in _measure_notes(
            measures[source_index], divisions
        ):
            notes.append(
                ScoreNote(performed_index, offset, measure_len, pitch, string, fret)
            )
    return notes


def measure_count(path: Path) -> tuple[int, int]:
    """Return (written measures, performed measures) for a score."""
    root = ET.parse(path).getroot()
    part = _tablature_part(root)
    measures = part.findall("measure")
    return len(measures), len(unfold_repeats(measures))


def read_tuning(path: Path) -> dict[int, int]:
    """Read the declared open-string tuning from ``<staff-tuning>``.

    GAPS scores state their tuning explicitly, which matters because roughly a
    quarter of the dataset is dropped-D or another scordatura. Beware the
    numbering: ``<staff-tuning line>`` counts 1 = lowest-pitched string, the
    opposite of ``<string>``, which counts 1 = highest. This returns the
    ``<string>`` convention so it composes with :func:`string_to_index`.

    Returns an empty mapping when the score omits the element, leaving the
    caller to fall back on :func:`infer_tuning`.
    """
    root = ET.parse(path).getroot()
    tuning: dict[int, int] = {}
    for element in root.findall(".//staff-details/staff-tuning"):
        line = element.get("line")
        step = element.findtext("tuning-step")
        octave = element.findtext("tuning-octave")
        if not (line and step and octave):
            continue
        alter = int(float(element.findtext("tuning-alter") or 0))
        pitch = 12 * (int(octave) + 1) + _STEP_SEMITONES[step] + alter
        tuning[N_STRINGS + 1 - int(line)] = pitch
    return tuning


def infer_tuning(notes: list[ScoreNote]) -> dict[int, int]:
    """Infer each string's open-string pitch from the score's own tablature.

    Used when a score omits ``<staff-tuning>``. ``pitch - fret`` is the open
    pitch of the string a note is played on, so the tuning falls out of the
    tablature itself with no assumption of standard tuning.

    Returns a mapping of MusicXML string number → open MIDI pitch, using the
    most common value per string so a stray mis-notated fret cannot shift it.
    """
    tally: dict[int, dict[int, int]] = {}
    for note in notes:
        counts = tally.setdefault(note.string, {})
        open_pitch = note.pitch - note.fret
        counts[open_pitch] = counts.get(open_pitch, 0) + 1
    return {
        string: max(counts.items(), key=lambda kv: kv[1])[0]
        for string, counts in tally.items()
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _tablature_part(root: ET.Element) -> ET.Element | None:
    """The part whose notes carry technical string annotations."""
    for part in root.findall("part"):
        if part.find(".//technical/string") is not None:
            return part
    return None


def _scan_attributes(measures: list[ET.Element]) -> dict[int, tuple[int, int]]:
    """Resolve (divisions, measure length in divisions) for every measure.

    Both are stateful in MusicXML — an ``<attributes>`` element applies until
    the next one — so this walks the measures in document order once and
    records the value in force at each.
    """
    divisions, beats, beat_type = 1, 4, 4
    resolved = {}
    for index, measure in enumerate(measures):
        for attributes in measure.findall("attributes"):
            value = attributes.findtext("divisions")
            if value:
                divisions = int(value)
            time = attributes.find("time")
            if time is not None:
                beats = int(time.findtext("beats"))
                beat_type = int(time.findtext("beat-type"))
        resolved[index] = (divisions, int(divisions * beats * 4 / beat_type))
    return resolved


def unfold_repeats(measures: list[ET.Element]) -> list[int]:
    """Expand repeat barlines into the sequence of measures actually performed.

    Returns source measure indices in performance order. Syncpoint indices are
    performed measure numbers, so this expansion is what makes the two line up;
    the caller cross-checks its length against the syncpoint count.

    Handles forward/backward repeat barlines and voltas. Da capo and dal segno
    jumps are not expanded — they are absent from the GAPS scores inspected,
    and the syncpoint cross-check flags any score where that assumption fails
    rather than letting a wrong expansion through silently.
    """
    order: list[int] = []
    index = 0
    section_start = 0
    pass_number = 1

    while index < len(measures):
        measure = measures[index]

        for barline in measure.findall("barline"):
            repeat = barline.find("repeat")
            if (
                repeat is not None
                and repeat.get("direction") == "forward"
                and barline.get("location") != "right"
                and index != section_start
            ):
                # Entering a new repeated section restarts the pass count.
                section_start = index
                pass_number = 1

        # A volta belonging to other passes is skipped wholesale. The pass
        # counter has to be tracked explicitly rather than derived from the
        # repeat barline: the second-time volta usually has no backward
        # repeat of its own, so there is nothing there to count.
        endings = _ending_numbers(measure)
        if endings and pass_number not in endings:
            index = _skip_ending(measures, index)
            continue

        order.append(index)

        jumped = False
        for barline in measure.findall("barline"):
            repeat = barline.find("repeat")
            if repeat is not None and repeat.get("direction") == "backward":
                if pass_number < int(repeat.get("times", 2)):
                    pass_number += 1
                    index = section_start
                    jumped = True
                else:
                    # Section finished; a later one starts counting afresh.
                    pass_number = 1
                break

        if not jumped:
            index += 1

        # Guard against a malformed repeat structure looping forever.
        if len(order) > 20 * len(measures) + 100:
            return list(range(len(measures)))

    return order


def _ending_numbers(measure: ET.Element) -> set[int]:
    """Volta numbers a measure belongs to, empty if it is not in a volta."""
    numbers: set[int] = set()
    for barline in measure.findall("barline"):
        ending = barline.find("ending")
        if ending is not None and ending.get("type") == "start":
            for part in (ending.get("number") or "").split(","):
                part = part.strip()
                if part.isdigit():
                    numbers.add(int(part))
    return numbers


def _skip_ending(measures: list[ET.Element], index: int) -> int:
    """First measure after the volta block starting at `index`."""
    for candidate in range(index, len(measures)):
        for barline in measures[candidate].findall("barline"):
            ending = barline.find("ending")
            if ending is not None and ending.get("type") in ("stop", "discontinue"):
                return candidate + 1
    return len(measures)


def _measure_notes(
    measure: ET.Element, divisions: int
) -> list[tuple[int, int, int, int]]:
    """Sounding notes of one measure as (offset, pitch, string, fret).

    Tracks the MusicXML cursor through ``<backup>``/``<forward>`` so multi-voice
    measures place notes at their true offsets rather than end to end. Chord
    members repeat the previous note's onset, and grace notes take the onset of
    the note they decorate without advancing the cursor.
    """
    notes: list[tuple[int, int, int, int]] = []
    cursor = 0
    previous_onset = 0

    for element in measure:
        if element.tag == "backup":
            cursor -= int(float(element.findtext("duration", "0") or 0))
        elif element.tag == "forward":
            cursor += int(float(element.findtext("duration", "0") or 0))
        elif element.tag == "note":
            duration = int(float(element.findtext("duration", "0") or 0))
            is_chord = element.find("chord") is not None
            is_grace = element.find("grace") is not None
            pitch_element = element.find("pitch")

            if element.find("rest") is not None or pitch_element is None:
                if not is_chord and not is_grace:
                    cursor += duration
                continue

            onset = previous_onset if is_chord else cursor
            string = element.findtext(".//technical/string")
            fret = element.findtext(".//technical/fret")
            if string is not None and fret is not None:
                notes.append(
                    (max(0, onset), _midi_pitch(pitch_element), int(string), int(fret))
                )

            previous_onset = onset
            if not is_chord and not is_grace:
                cursor += duration

    return notes


def _midi_pitch(pitch: ET.Element) -> int:
    """MIDI note number from a MusicXML <pitch> element."""
    step = _STEP_SEMITONES[pitch.findtext("step")]
    octave = int(pitch.findtext("octave"))
    alter = int(float(pitch.findtext("alter") or 0))
    return 12 * (octave + 1) + step + alter
