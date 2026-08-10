"""Flatten the pitch drop at the end of a note.

A plucked string falls slightly in pitch as it dies: the finger lifts, tension
drops, and the frequency follows. The effect is real and GuitarSet's labels
record it faithfully — 21% of notes end more than 50 cents below their own
body.

It should not be reported. The transcription target is MIDI, where a falling
pitch at a note end is a pitch bend, and a model trained on labels that contain
one will learn to emit a bend every time a note stops. What the drop actually
signals is that the note is ending, which belongs to onset/offset detection
rather than to the pitch channel.

So this module holds the pre-drop pitch through the drop, making the label read
as if the string never slackened. That is a deliberate departure from acoustic
ground truth, taken because the downstream task wants the note that was played
rather than the physics of it decaying.

Two gestures are genuinely descending and must survive: a slide down the
fretboard, and a bend released back to the fretted note while it still rings.
Neither re-plucks the string, so both live inside a single annotated note and
can fall in its final frames. They are separated from decay by **duration** —
see :func:`classify_drop`.

Nothing here needs anything beyond GuitarSet's own annotations.
"""
import numpy as np

# Measured over GuitarSet's 63,370 notes: a decay drop lasts a median of 23 ms
# and 90% finish inside 64 ms — four frames at the native 5.8 ms grid. No
# gesture is possible in that time. Moving a finger along a fretboard, or
# relaxing a bend, takes on the order of 100 ms, which is what makes duration a
# usable discriminator where pitch and loudness are not: end-of-note amplitude
# is 0.176 of body for a stable drop and 0.161 for an unstable one, and a
# neural tracker's periodicity reads 0.530 against 0.538 — neither separates.
DECAY_MAX_MS   = 60.0
GESTURE_MIN_MS = 100.0

# How far BELOW the body a frame must sit to count as part of the drop.
# Downward only: a string losing tension cannot rise, so an upward excursion at
# a note end is a different phenomenon and is left alone. A signed test also
# avoids reading a note whose body drifts upward as departing from its own
# median, which would swallow most of the note.
DROP_THRESHOLD_CENTS = 25.0

# A gesture arrives somewhere and stays; a note dying just keeps falling. This
# is the spread allowed across the final frames for a long drop to read as a
# slide or bend release.
GESTURE_STABLE_CENTS = 25.0

DROP_ACTIONS = ("none", "flattened", "ambiguous-flattened", "preserved")


def flatten_tails(
    pitch: np.ndarray,
    note_ids: np.ndarray,
    hop_seconds: float,
    threshold_cents: float = DROP_THRESHOLD_CENTS,
) -> tuple[np.ndarray, dict[str, int]]:
    """Hold the pre-drop pitch through each note's closing pitch drop.

    Parameters
    ----------
    pitch : (6, n_frames) float32
        Hz, 0 for unvoiced.
    note_ids : (6, n_frames) int32
        Per-frame note index, -1 where unvoiced, as produced by
        :func:`preprocess.extract_pitch_note_arrays_jams`. Exact note
        boundaries are what make this possible at all — a drop is defined
        relative to its own note's body, and two notes run together would
        pollute that reference.
    hop_seconds:
        Seconds per frame. The thresholds here are durations, so this is what
        converts them to frame counts.

    Returns
    -------
    pitch  : new array; the input is not modified
    counts : how many notes fell into each of DROP_ACTIONS

    Voicing is untouched — only pitch values inside already-voiced frames
    change, so the note's extent is exactly as annotated.
    """
    pitch = pitch.copy()
    counts = {action: 0 for action in DROP_ACTIONS}

    for s in range(pitch.shape[0]):
        ids = note_ids[s]
        for note in np.unique(ids[ids >= 0]):
            frames = np.where(ids == note)[0]
            contour = pitch[s, frames]
            sounding = contour[contour > 0]
            if sounding.size == 0:
                counts["none"] += 1
                continue

            run = _detect_with_refined_reference(contour, threshold_cents)
            action = classify_drop(
                run * hop_seconds * 1000.0, _cents_of_tail(contour, run)
            )
            counts[action] += 1

            if action in ("flattened", "ambiguous-flattened") and run:
                target = frames[frames.size - run:]
                pitch[s, target] = np.float32(_flatten_value(contour, run))

    return pitch, counts


def _detect_with_refined_reference(
    contour: np.ndarray, threshold_cents: float
) -> int:
    """Find the drop, then re-find it against a reference that excludes it.

    The note's median is a good stand-in for its body only while the drop is a
    small share of the note. One refinement pass removes that dependence, so no
    arbitrary 'the body is the first 80%' fraction has to be chosen: detect
    once against the whole-note median, drop those frames, take the median of
    what remains, and detect again.
    """
    sounding = contour[contour > 0]
    if sounding.size == 0:
        return 0

    first = detect_drop(contour, float(np.median(sounding)), threshold_cents)
    body = contour[: contour.size - first]
    body = body[body > 0]
    if body.size == 0:
        return first
    return detect_drop(contour, float(np.median(body)), threshold_cents)


def detect_drop(
    contour: np.ndarray,
    ref: float,
    threshold_cents: float = DROP_THRESHOLD_CENTS,
) -> int:
    """Length of the trailing run that has fallen below the body pitch.

    Walks backward from the final frame while the pitch stays more than
    ``threshold_cents`` *below* ``ref``. Zero means the note does not end flat
    and there is nothing to explain — including when it ends sharp, which is
    not this phenomenon.

    Detected per note rather than assumed, because a drop's length varies from
    one frame to eleven. A fixed window is wrong at both ends: too short and it
    misses the quarter of drops beginning before the final 20%, too long and on
    a half-second note it swallows a genuine bend release.
    """
    if ref <= 0 or contour.size == 0 or not (contour > 0).any():
        return 0

    run = 0
    for value in contour[::-1]:
        if value <= 0:
            break
        if 1200.0 * np.log2(value / ref) >= -threshold_cents:
            break
        run += 1
    return run


def classify_drop(duration_ms: float, tail_cents: np.ndarray) -> str:
    """Decide whether a drop is a dying note or a played gesture.

    Duration is the discriminator; see DECAY_MAX_MS. A long fall must also
    *settle* to count as a gesture — a slide or a released bend arrives at a
    pitch and holds it there while the note keeps ringing.

    The ambiguous middle is flattened rather than preserved, and labelled so.
    The costs are asymmetric: wrongly preserving a decay writes a pitch bend
    into the labels that a model will learn to reproduce, which is worse for a
    MIDI target than wrongly flattening a gesture.
    """
    if duration_ms <= 0:
        return "none"
    if duration_ms < DECAY_MAX_MS:
        return "flattened"

    settled = (
        tail_cents.size >= 3
        and np.ptp(tail_cents[-3:]) < GESTURE_STABLE_CENTS
    )
    if duration_ms >= GESTURE_MIN_MS and settled:
        return "preserved"
    return "ambiguous-flattened"


def _flatten_value(contour: np.ndarray, run: int) -> float:
    """The pitch to hold — 'as if it never dropped'.

    Taken from the frames immediately before the drop rather than from the
    note's median, so the held value continues whatever the note was actually
    doing (a slow vibrato, a settled bend) and leaves no step where the
    flattening begins.
    """
    before = contour[: contour.size - run]
    before = before[before > 0]
    if before.size == 0:
        sounding = contour[contour > 0]
        return float(np.median(sounding)) if sounding.size else 0.0
    return float(np.median(before[-5:]))


def _cents_of_tail(contour: np.ndarray, run: int) -> np.ndarray:
    """The drop's own frames, in cents relative to the pitch it left."""
    if run == 0:
        return np.empty(0)
    value = _flatten_value(contour, run)
    tail = contour[contour.size - run:]
    out = np.full(tail.shape, np.nan)
    ok = (tail > 0) & (value > 0)
    out[ok] = 1200.0 * np.log2(tail[ok] / value)
    return out


def format_counts(counts: dict[str, int]) -> str:
    """One line describing what flattening did."""
    total = sum(counts.values())
    if total == 0:
        return "  no notes to flatten"
    return "  " + "   ".join(
        f"{action}={counts[action]} ({counts[action] / total:.1%})"
        for action in DROP_ACTIONS
        if counts[action]
    )
