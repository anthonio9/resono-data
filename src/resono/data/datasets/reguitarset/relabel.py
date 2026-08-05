"""Rewrite note tails from FCNF0++ estimates, and record what changed.

GuitarSet's labels are pYIN with manual correction. Their note bodies are
sound; their tails — the decaying end of a plucked note — drift and jump
octaves, because that is where a tracker running on the polyphonic mic mix has
least signal to work with. This module keeps the human bodies and hands the
tails to FCNF0++ running on the isolated string.

There is deliberately **no gate**: FCNF0++ is assumed better and owns every
tail. Where it disagrees with the human labels on the *body* — the part both
should agree on — the disagreement is measured and written to the audit trail
rather than acted on, so the assumption can be checked by hand afterwards
instead of being quietly enforced here.
"""
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

# Cents between two pitches an octave apart. Half of this is the fold
# threshold: past 600 cents the nearer interpretation is the neighbouring
# octave, which is exactly the error being corrected.
OCTAVE_CENTS = 1200.0

TAIL_POLICIES   = ("track", "hold")
OFFSET_POLICIES = ("both", "trim", "extend", "none")


@dataclass
class NoteAudit:
    """One row per note: what the two sources said, and what was done."""

    stem: str
    string: int
    note: int
    t_start: float
    t_end: float
    n_frames: int
    human_body_hz: float
    fcnf0_body_hz: float
    body_cents_diff: float
    body_periodicity: float
    human_tail_hz_median: float
    fcnf0_tail_hz_median: float
    tail_cents_change_median: float
    tail_cents_change_max: float
    octave_folds: int
    frames_trimmed: int
    frames_extended: int


AUDIT_COLUMNS = [f.name for f in fields(NoteAudit)]


def relabel_tails(
    pitch: np.ndarray,
    voiced: np.ndarray,
    note_ids: np.ndarray,
    f0: np.ndarray,
    periodicity: np.ndarray,
    stem: str,
    hop_seconds: float,
    tail_policy: str = "track",
    offset_policy: str = "both",
    divider: int = 5,
    periodicity_threshold: float = 0.3,
    max_extend_frames: int = 64,
    median_filter: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[NoteAudit]]:
    """Replace each note's tail using the FCNF0++ contour for that string.

    Parameters
    ----------
    pitch, voiced, note_ids:
        Labels on the target grid, as produced by
        ``guitarset.preprocess.extract_pitch_note_arrays_jams``. ``note_ids``
        carries exact note boundaries — including a re-struck same-pitch note
        that a voicing transition would miss — and every decision here is made
        per note, so it is what makes the surgery possible at all.
    f0, periodicity:
        FCNF0++ output for the same grid, (6, n_frames).
    tail_policy:
        'track' (default) takes FCNF0++'s tail contour, folded back toward the
        note body's octave. 'hold' instead extrapolates the body's own pitch
        across the tail, on the reasoning that a plucked string decays in
        amplitude rather than frequency, leaving FCNF0++ to contribute only the
        offset. 'hold' is the more conservative of the two.
    offset_policy:
        Which direction periodicity is allowed to move the note end. 'both'
        (default) trims frames whose periodicity has fallen below threshold and
        extends into frames still ringing above it; 'trim' and 'extend' allow
        one direction only; 'none' leaves voicing exactly as labelled and
        rewrites pitch values alone.
    divider:
        The last 1/divider of a note is its tail. Default 5, i.e. the final 20%.
    periodicity_threshold:
        Below this, FCNF0++ considers the string no longer sounding.
    max_extend_frames:
        Ceiling on how far a note may be extended, independent of periodicity.
    median_filter:
        Odd window for smoothing the tail under 'track'. **Off by default**,
        and deliberately so: penn already Viterbi-decodes the posteriorgram,
        which is what suppresses impulsive outliers, and per-string fmin/fmax
        plus :func:`octave_fold` cover the octave errors. What is left in a
        decaying tail is slow drift, which a median filter does not touch. The
        knob stays available for when the audit shows jitter worth removing —
        relabelling is the cheap stage, so turning it on is one re-run.

    Returns
    -------
    pitch, voiced : new arrays (inputs are not modified)
    audits        : one NoteAudit per note, in (string, note) order
    """
    if tail_policy not in TAIL_POLICIES:
        raise ValueError(f"tail_policy must be one of {TAIL_POLICIES}")
    if offset_policy not in OFFSET_POLICIES:
        raise ValueError(f"offset_policy must be one of {OFFSET_POLICIES}")

    pitch  = pitch.copy()
    voiced = voiced.copy()
    audits: list[NoteAudit] = []

    n_strings, n_frames = pitch.shape
    may_trim   = offset_policy in ("both", "trim")
    may_extend = offset_policy in ("both", "extend")

    for s in range(n_strings):
        ids = note_ids[s]
        # Frames already spoken for by *some* note on this string. Extension
        # must not write into a neighbouring note, and the original voicing is
        # the record of where those notes are — so it is captured before any
        # of this string's notes are edited.
        claimed = ids >= 0

        for note in np.unique(ids[ids >= 0]):
            frames = np.where(ids == note)[0]
            audit = _relabel_note(
                pitch, voiced, f0, periodicity, claimed,
                s, int(note), frames, n_frames, stem, hop_seconds,
                tail_policy, may_trim, may_extend,
                divider, periodicity_threshold, max_extend_frames, median_filter,
            )
            if audit is not None:
                audits.append(audit)

    return pitch, voiced, audits


def _relabel_note(
    pitch, voiced, f0, periodicity, claimed,
    s, note, frames, n_frames, stem, hop_seconds,
    tail_policy, may_trim, may_extend,
    divider, periodicity_threshold, max_extend_frames, median_filter,
) -> NoteAudit | None:
    """Rewrite one note's tail in place; return its audit row."""
    n = frames.size
    n_tail = n // divider

    # Split body from tail. A note shorter than `divider` frames has no tail
    # under this definition; it still gets its offset examined, since a very
    # short note is exactly the kind pYIN may have cut off early.
    body_frames = frames[: n - n_tail]
    tail_frames = frames[n - n_tail:]
    if body_frames.size == 0:
        return None

    body = pitch[s, body_frames]
    body = body[body > 0]
    if body.size == 0:
        return None
    ref = float(np.median(body))

    fcnf0_body = f0[s, body_frames]
    fcnf0_body = fcnf0_body[fcnf0_body > 0]
    fcnf0_body_hz = float(np.median(fcnf0_body)) if fcnf0_body.size else 0.0
    body_cents_diff = _cents(fcnf0_body_hz, ref) if fcnf0_body_hz > 0 else float("nan")

    human_tail = pitch[s, tail_frames]
    human_tail_median = (
        float(np.median(human_tail[human_tail > 0])) if (human_tail > 0).any() else 0.0
    )
    fcnf0_tail = f0[s, tail_frames]
    fcnf0_tail_median = (
        float(np.median(fcnf0_tail[fcnf0_tail > 0])) if (fcnf0_tail > 0).any() else 0.0
    )

    extend_frames = _extension(
        periodicity, claimed, s, frames[-1], n_frames,
        periodicity_threshold, max_extend_frames,
    ) if may_extend else np.empty(0, dtype=np.int64)

    trim_frames = _trim(
        periodicity, s, tail_frames, periodicity_threshold,
    ) if may_trim else np.empty(0, dtype=np.int64)

    write_frames = np.concatenate([tail_frames, extend_frames]).astype(np.int64)
    octave_folds = 0

    if write_frames.size:
        if tail_policy == "track":
            new_pitch, octave_folds = _tracked(
                f0[s, write_frames], ref, median_filter
            )
            # penn always returns a pitch, so a zero here can only come from
            # the pad in f0._fit — past the end of what was tracked. Fall back
            # to the body rather than writing a hole into the labels.
            missing = new_pitch <= 0
            if missing.any():
                new_pitch[missing] = _held(write_frames, body_frames, pitch[s], ref)[missing]
        else:
            new_pitch = _held(write_frames, body_frames, pitch[s], ref)

        pitch[s, write_frames]  = new_pitch.astype(np.float32)
        voiced[s, write_frames] = True

    # Trimming last: a frame can be both written and then silenced, and the
    # silence is what must survive.
    if trim_frames.size:
        pitch[s, trim_frames]  = 0.0
        voiced[s, trim_frames] = False

    tail_change = _tail_change(pitch[s], human_tail, tail_frames)

    return NoteAudit(
        stem=stem,
        string=s,
        note=note,
        t_start=float(frames[0] * hop_seconds),
        t_end=float((frames[-1] + 1) * hop_seconds),
        n_frames=int(n),
        human_body_hz=ref,
        fcnf0_body_hz=fcnf0_body_hz,
        body_cents_diff=body_cents_diff,
        body_periodicity=float(np.median(periodicity[s, body_frames])),
        human_tail_hz_median=human_tail_median,
        fcnf0_tail_hz_median=fcnf0_tail_median,
        tail_cents_change_median=tail_change[0],
        tail_cents_change_max=tail_change[1],
        octave_folds=int(octave_folds),
        frames_trimmed=int(trim_frames.size),
        frames_extended=int(extend_frames.size),
    )


# ---------------------------------------------------------------------------
# Tail pitch
# ---------------------------------------------------------------------------

def _tracked(values: np.ndarray, ref: float, median_filter: int):
    """FCNF0++'s contour, folded into the body's octave and smoothed."""
    folded, folds = octave_fold(values, ref)
    return _median_filter(folded, median_filter), folds


def octave_fold(values: np.ndarray, ref: float) -> tuple[np.ndarray, int]:
    """Move each value to the octave nearest ``ref``.

    An octave error is a tracker latching onto a harmonic or a subharmonic: the
    pitch class is right and only the register is wrong. Since a decaying
    string does not actually change octave, any whole-octave departure from the
    note body is an error with an obvious correction, and folding it back
    recovers a usable estimate where discarding the frame would not.
    Deviations short of half an octave are left alone — those are real.
    """
    out = values.astype(np.float64).copy()
    usable = out > 0
    if not usable.any() or ref <= 0:
        return out, 0

    shifts = np.zeros(out.shape, dtype=np.float64)
    shifts[usable] = np.round(
        1200.0 * np.log2(out[usable] / ref) / OCTAVE_CENTS
    )
    out[usable] = out[usable] * 2.0 ** (-shifts[usable])
    return out, int((shifts != 0).sum())


def _held(
    write_frames: np.ndarray,
    body_frames: np.ndarray,
    string_pitch: np.ndarray,
    ref: float,
) -> np.ndarray:
    """Extrapolate the body's own pitch across the given frames.

    Fitted in log-frequency, where a constant musical interval is a constant
    distance, so the fit is not dominated by the high strings.

    The extrapolation is deliberately unbounded. A bend is part of the note —
    a whole-tone bend is two hundred cents and still the same note event, with
    no new onset — so capping the excursion would flatten real playing and put
    a discontinuity at the body/tail boundary. Nor is a bound needed: the tail
    is at most 1/divider of the note, so continuing a whole-body least-squares
    trend across it adds at most that fraction of the body's own excursion.
    The geometry bounds it without help.
    """
    usable = body_frames[string_pitch[body_frames] > 0]
    # A single point has no slope to fit; hold the reference flat instead.
    if usable.size < 2:
        return np.full(write_frames.shape, ref, dtype=np.float64)

    slope, intercept = np.polyfit(usable, np.log2(string_pitch[usable]), 1)
    return np.exp2(slope * write_frames + intercept)


def _median_filter(values: np.ndarray, size: int) -> np.ndarray:
    """Sliding median, edge-padded so the output keeps its length.

    The window is clamped to the input and then forced odd. Both steps matter:
    a tail can be shorter than the requested window, and an even window would
    make the padding asymmetric and return one value too many.
    """
    if size <= 1 or values.size < 2:
        return values

    size = min(size, values.size)
    if size % 2 == 0:
        size -= 1
    if size <= 1:
        return values

    pad = size // 2
    padded = np.pad(values, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    return np.median(windows, axis=-1)


# ---------------------------------------------------------------------------
# Note offsets
# ---------------------------------------------------------------------------

def _extension(
    periodicity, claimed, s, last_frame, n_frames, threshold, max_frames
) -> np.ndarray:
    """Frames past the labelled end where the string is still ringing.

    Stops at the first frame below threshold, at any frame another note already
    claims, and at max_frames — so a note can never swallow its successor, and
    a periodicity contour that never quite decays cannot run to the end of the
    track.
    """
    out = []
    frame = last_frame + 1
    while (
        frame < n_frames
        and len(out) < max_frames
        and not claimed[frame]
        and periodicity[s, frame] >= threshold
    ):
        out.append(frame)
        frame += 1
    return np.asarray(out, dtype=np.int64)


def _trim(periodicity, s, tail_frames, threshold) -> np.ndarray:
    """Trailing tail frames whose periodicity has fallen below threshold.

    Walks backward from the labelled end and stops at the first frame still
    sounding, so only a contiguous run at the very end is removed. A dip in the
    middle of a tail is left alone — that is a tracker artefact, not the note
    ending twice.
    """
    if tail_frames.size == 0:
        return np.empty(0, dtype=np.int64)

    cut = tail_frames.size
    while cut > 0 and periodicity[s, tail_frames[cut - 1]] < threshold:
        cut -= 1
    return tail_frames[cut:]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _cents(value: float, reference: float) -> float:
    """Signed interval from reference to value, in cents."""
    if value <= 0 or reference <= 0:
        return float("nan")
    return float(1200.0 * np.log2(value / reference))


def _tail_change(string_pitch, human_tail, tail_frames) -> tuple[float, float]:
    """How far the tail moved, in cents: (median, max absolute)."""
    if tail_frames.size == 0:
        return float("nan"), float("nan")

    new_tail = string_pitch[tail_frames]
    both = (new_tail > 0) & (human_tail > 0)
    if not both.any():
        return float("nan"), float("nan")

    change = 1200.0 * np.log2(new_tail[both] / human_tail[both])
    return float(np.median(change)), float(np.abs(change).max())


# ---------------------------------------------------------------------------
# Audit output
# ---------------------------------------------------------------------------

def write_audit(rows: list[NoteAudit], path: Path) -> None:
    """Write the per-note audit trail as CSV."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarise(rows: list[NoteAudit]) -> dict:
    """Aggregate the audit trail into the numbers worth reading first.

    The body-disagreement figures are the headline: FCNF0++ and the human
    labels are both supposed to be right about note bodies, so the size of that
    gap is the best available evidence for whether trusting FCNF0++ with the
    tails was justified. A body_cents_diff beyond 600 means the two disagree by
    more than half an octave — almost certainly an octave error by one of them,
    and worth looking at individually.
    """
    if not rows:
        return {"notes": 0}

    body_diff = np.array([r.body_cents_diff for r in rows], dtype=np.float64)
    body_diff = body_diff[np.isfinite(body_diff)]
    tail_change = np.array([r.tail_cents_change_median for r in rows], dtype=np.float64)
    tail_change = tail_change[np.isfinite(tail_change)]
    trimmed  = np.array([r.frames_trimmed for r in rows])
    extended = np.array([r.frames_extended for r in rows])
    periodicity = np.array([r.body_periodicity for r in rows], dtype=np.float64)

    return {
        "notes": len(rows),
        "tracks": len({r.stem for r in rows}),
        "body_disagreement_cents": _distribution(np.abs(body_diff)),
        "body_disagreement_over_50c":  _fraction(np.abs(body_diff) > 50.0),
        "body_disagreement_over_600c": _fraction(np.abs(body_diff) > 600.0),
        "body_periodicity": _distribution(periodicity),
        "tail_change_cents": _distribution(np.abs(tail_change)),
        "octave_folds": int(sum(r.octave_folds for r in rows)),
        "notes_folded": _fraction(np.array([r.octave_folds > 0 for r in rows])),
        "frames_trimmed": _distribution(trimmed.astype(np.float64)),
        "frames_extended": _distribution(extended.astype(np.float64)),
        "notes_trimmed":  _fraction(trimmed > 0),
        "notes_extended": _fraction(extended > 0),
    }


def _distribution(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"median": None, "p95": None, "max": None}
    return {
        "median": round(float(np.median(values)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
        "max": round(float(values.max()), 4),
    }


def _fraction(mask: np.ndarray) -> float:
    return round(float(mask.mean()), 4) if mask.size else 0.0


def format_summary(summary: dict) -> str:
    """Render summarise() output as a short readable block."""
    if not summary.get("notes"):
        return "No notes were relabelled."

    def dist(key: str) -> str:
        d = summary[key]
        if d["median"] is None:
            return "n/a"
        return f"median {d['median']:>8.2f}   p95 {d['p95']:>9.2f}   max {d['max']:>10.2f}"

    return "\n".join([
        f"  notes                    {summary['notes']} across {summary['tracks']} tracks",
        f"  body disagreement (¢)    {dist('body_disagreement_cents')}",
        f"    beyond 50¢             {summary['body_disagreement_over_50c']:.1%} of notes",
        f"    beyond 600¢            {summary['body_disagreement_over_600c']:.1%} of notes"
        "   (suspected octave error)",
        f"  body periodicity         {dist('body_periodicity')}",
        f"  tail change (¢)          {dist('tail_change_cents')}",
        f"  octave folds             {summary['octave_folds']} frames"
        f" in {summary['notes_folded']:.1%} of notes",
        f"  frames trimmed           {dist('frames_trimmed')}"
        f"   ({summary['notes_trimmed']:.1%} of notes)",
        f"  frames extended          {dist('frames_extended')}"
        f"   ({summary['notes_extended']:.1%} of notes)",
    ])
