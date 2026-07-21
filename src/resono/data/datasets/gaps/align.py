"""Transfer string labels from the GAPS score onto performance timing.

The three GAPS annotation sources are complementary and none is sufficient
alone:

* the MusicXML carries a string and fret for every note, but only score time;
* the fine-aligned MIDI carries precise performance onsets, but every note is
  on channel 0 — it holds no string information at all;
* the syncpoints map performed measures to seconds, but only at measure
  granularity, so within-measure timing is linear and unreliable under rubato.

So the score's labels have to be carried onto the MIDI's timeline. Alignment
by note order alone fails: performers take repeats the score writes once, and
chord members are ordered differently on each side, which caps a pure sequence
alignment well below full coverage. Alignment by interpolated time alone fails
too, on pieces whose syncpoints are sparse. This module combines them —
a monotonic alignment whose match reward decays with the syncpoint-predicted
time error — which recovers substantially more of both cases than either.

Note on label quality: GAPS has no per-string ground truth, so this
assignment cannot be validated for accuracy against the dataset itself. The
dataset's own ``.match`` files pair only a fifth to a quarter of performed
notes and are not usable as a reference. Treat string labels here as
score-derived and approximate, and watch string_assignment_accuracy on a
dataset that does have ground truth to detect whether they hurt.
"""
import numpy as np

from resono.data.datasets.gaps.score import ScoreNote, string_to_index

_NEG = -1.0e18   # stands in for -inf without producing NaNs in the DP

# Highest fret considered when falling back to a heuristic string choice.
_MAX_FRET = 19


def predict_times(notes: list[ScoreNote], syncpoints: list) -> np.ndarray:
    """Predict a performance time for every score note from the syncpoints.

    Syncpoints are ``[performed_measure, seconds]`` or, where the annotators
    refined a measure internally, ``[performed_measure, seconds, offset]`` with
    the offset given in divisions. Both forms become anchors on a fractional
    measure axis, and note positions are linearly interpolated between them.

    The result is only as good as the anchor spacing — it is a prior for the
    alignment, not a source of onsets.
    """
    measure_len = {note.measure: note.measure_len for note in notes}

    anchor_positions, anchor_times = [], []
    for point in syncpoints:
        if len(point) < 2:
            continue
        measure, seconds = point[0], point[1]
        offset = point[2] if len(point) > 2 else 0
        length = measure_len.get(measure) or 1
        anchor_positions.append(measure + offset / length)
        anchor_times.append(seconds)

    if len(anchor_positions) < 2:
        raise ValueError("need at least two syncpoints to interpolate time")

    positions = np.asarray(anchor_positions, dtype=np.float64)
    times = np.asarray(anchor_times, dtype=np.float64)
    order = np.argsort(positions, kind="stable")
    positions, times = positions[order], times[order]

    # np.interp needs strictly increasing x; duplicate anchors keep the first.
    unique, first = np.unique(positions, return_index=True)
    return np.interp(
        np.array([note.position for note in notes], dtype=np.float64),
        unique,
        times[first],
    )


def align(
    midi_notes: list[tuple[float, float, int]],
    score_notes: list[ScoreNote],
    predicted_times: np.ndarray,
    tolerance: float = 2.0,
    gap_penalty: float = -0.3,
    chord_window: float = 0.05,
) -> dict[int, int]:
    """Match MIDI notes to score notes, returning {midi index: score index}.

    A note pair may only match when the pitches are equal and the MIDI onset
    falls within `tolerance` seconds of the score note's predicted time; the
    reward decays linearly with that error, so among equally plausible score
    notes the alignment prefers the one the syncpoints point at. Monotonicity
    is enforced by the dynamic program, which is what stops a note from being
    matched to a same-pitch note elsewhere in the piece.

    Both sequences are sorted by (time, pitch) first. Without that, chord
    members — simultaneous, and ordered by neither source consistently —
    break the monotonicity the alignment depends on.

    MIDI onsets are quantised into `chord_window` groups before that sort.
    A performer rolls a chord, so its notes arrive milliseconds apart and
    would otherwise sort by roll direction, while the score holds them at one
    onset and sorts them by pitch. Any chord not rolled bottom-up then breaks
    monotonicity. Quantising first lifts median coverage across the dataset
    from 90.7% to 98.0%; the window has to stay well under a fast note value,
    hence 50 ms rather than something larger.
    """
    if not midi_notes or not score_notes:
        return {}

    onset_group = _group_onsets([note[0] for note in midi_notes], chord_window)
    midi_order = sorted(
        range(len(midi_notes)), key=lambda i: (onset_group[i], midi_notes[i][2])
    )
    score_order = sorted(
        range(len(score_notes)),
        key=lambda j: (predicted_times[j], score_notes[j].pitch),
    )

    midi_times = np.array([midi_notes[i][0] for i in midi_order])
    midi_pitch = np.array([midi_notes[i][2] for i in midi_order])
    score_times = np.array([predicted_times[j] for j in score_order])
    score_pitch = np.array([score_notes[j].pitch for j in score_order])

    # float64 throughout: the traceback re-derives which case produced each
    # cell by comparing sums, and float32 rounding on a few thousand
    # accumulated fractional rewards is enough to make that comparison pick
    # the wrong predecessor.
    n, m = len(midi_order), len(score_order)
    table = np.empty((n + 1, m + 1), dtype=np.float64)
    table[0] = np.arange(m + 1) * gap_penalty
    table[:, 0] = np.arange(n + 1) * gap_penalty

    columns = np.arange(m + 1) * gap_penalty
    for i in range(1, n + 1):
        rewards = _match_rewards(
            midi_pitch[i - 1], midi_times[i - 1], score_pitch, score_times, tolerance
        )
        # Best score ignoring same-row gaps, then a max-plus prefix scan folds
        # in "extend leftwards through gaps", keeping each row O(m).
        best = np.empty(m + 1, dtype=np.float64)
        best[0] = table[i - 1, 0] + gap_penalty
        best[1:] = np.maximum(table[i - 1, :-1] + rewards, table[i - 1, 1:] + gap_penalty)
        table[i] = np.maximum.accumulate(best - columns) + columns

    return _traceback(
        table, midi_order, score_order,
        midi_pitch, midi_times, score_pitch, score_times,
        tolerance, gap_penalty,
    )


def _group_onsets(onsets: list[float], window: float) -> list[float]:
    """Quantise onsets so notes within `window` of a group's start share a key.

    Groups start at the first note not already covered, rather than rounding
    to a fixed grid, so a chord is never split by falling either side of a
    grid line. Passing window <= 0 leaves the onsets untouched.
    """
    if window <= 0 or not onsets:
        return list(onsets)

    keys = []
    start = None
    for onset in onsets:                      # midi_notes arrive onset-sorted
        if start is None or onset - start > window:
            start = onset
        keys.append(start)
    return keys


def _match_rewards(
    pitch: int,
    time: float,
    score_pitch: np.ndarray,
    score_times: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Reward for matching one MIDI note against every score note."""
    error = np.abs(score_times - time)
    allowed = (score_pitch == pitch) & (error <= tolerance)
    return np.where(allowed, 1.0 - error / tolerance, _NEG)


def _traceback(
    table, midi_order, score_order,
    midi_pitch, midi_times, score_pitch, score_times,
    tolerance, gap_penalty,
) -> dict[int, int]:
    """Walk the filled table back to the matched pairs."""
    matches: dict[int, int] = {}
    i, j = len(midi_order), len(score_order)

    while i > 0 and j > 0:
        error = abs(score_times[j - 1] - midi_times[i - 1])
        reward = (
            1.0 - error / tolerance
            if midi_pitch[i - 1] == score_pitch[j - 1] and error <= tolerance
            else _NEG
        )
        if np.isclose(table[i, j], table[i - 1, j - 1] + reward, atol=1e-3):
            matches[midi_order[i - 1]] = score_order[j - 1]
            i -= 1
            j -= 1
        elif np.isclose(table[i, j], table[i - 1, j] + gap_penalty, atol=1e-3):
            i -= 1
        else:
            j -= 1

    return matches


def assign_strings(
    midi_notes: list[tuple[float, float, int]],
    score_notes: list[ScoreNote],
    matches: dict[int, int],
    tuning: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a string index for every MIDI note.

    Matched notes take the string the score assigns them. The remainder — the
    notes the alignment could not place — fall back to the lowest fret that is
    playable on a string not already sounding, using the tuning inferred from
    the score rather than assuming standard tuning.

    Returns
    -------
    strings   : int8  (n_notes,)  resono string index, 0 = low E
    from_score: bool  (n_notes,)  True where the string came from the score
    """
    n = len(midi_notes)
    strings = np.full(n, -1, dtype=np.int8)
    from_score = np.zeros(n, dtype=bool)

    for midi_index, score_index in matches.items():
        strings[midi_index] = string_to_index(score_notes[score_index].string)
        from_score[midi_index] = True

    # Committed intervals per string, so the fallback avoids putting two
    # simultaneous notes on one string — physically impossible on a guitar.
    busy: dict[int, list[tuple[float, float]]] = {s: [] for s in range(6)}
    for index in np.argsort([note[0] for note in midi_notes]):
        if strings[index] >= 0:
            onset, offset, _ = midi_notes[index]
            busy[int(strings[index])].append((onset, offset))

    open_pitches = {
        string_to_index(string): pitch for string, pitch in tuning.items()
    }

    for index in np.argsort([note[0] for note in midi_notes]):
        if strings[index] >= 0:
            continue
        onset, offset, pitch = midi_notes[index]

        candidates = []
        for string_index, open_pitch in open_pitches.items():
            fret = pitch - open_pitch
            if 0 <= fret <= _MAX_FRET:
                candidates.append((fret, string_index))
        if not candidates:
            continue
        candidates.sort()

        chosen = next(
            (
                string_index
                for _, string_index in candidates
                if not _overlaps(busy[string_index], onset, offset)
            ),
            candidates[0][1],
        )
        strings[index] = chosen
        busy[chosen].append((onset, offset))

    return strings, from_score


def _overlaps(intervals: list[tuple[float, float]], onset: float, offset: float) -> bool:
    """Whether [onset, offset) intersects any committed interval."""
    return any(onset < end and start < offset for start, end in intervals)
