"""Find where labelled notes really start in the audio.

Labels from annotation pipelines are often late: GuitarSet's note starts
sit about 17 ms after the pick attack, GOAT's wander by tens of ms from
track to track. This module locates the attack of each labelled note, so
the labels can be moved onto it (see labels.py). It works offline on the
whole track, so it may look ahead freely.

Two detectors, per labelled note (string, label frame, f0):

comb
    Energy in a Hann window (4 periods of f0, 10-46 ms) at the note's
    harmonics k * f0 (k = 1..8, below 0.45 * sample rate). Harmonics that
    coincide with a partial of a note attacked nearby on another string
    are left out, so a chord tone is timed by its own partials. Each
    harmonic's level is floored FLOOR_DB below its own peak: leakage from
    a louder note into a silent bin is not a rise.
    Coarse: the window end e maximising the mean over harmonics of
    log E_k(e) - log E_k(e - W), i.e. the note's partials appear between
    two adjacent windows. That mean rise, in dB, is the note's strength;
    below COMB_MIN_DB there is no comb estimate.
    Fine: the summed comb energy crosses halfway between the window before
    and the level after. A Hann window is half full when the attack sits at
    its middle, so attack = crossing - W / 2, whatever the loudness.

transient
    Broadband rise: log energy of the high-passed audio in 128-sample
    blocks (at 44.1 kHz) every quarter block; its peaks (scipy find_peaks)
    rising at least BB_MIN_DB. A pick click places the attack to about a
    millisecond.

Matching: notes whose labels lie within CLUSTER_MS of each other (any
string) form a cluster. Their comb estimates and the transient peaks are
matched one-to-one (scipy linear_sum_assignment, cost = distance, at most
MATCH_MS):
    "precise"    matched: the transient's time
    "comb"       unmatched: the comb time
    "transient"  a note alone in its cluster with no comb estimate (often a
                 re-strike, whose partials were already sounding) takes the
                 strongest transient in its window if it rises BB_ALONE_DB
    "none"       no estimate
    "kept"       a note in a multi-note cluster (a strum) while the track's
                 jitter is at most CLUSTER_SNAP_MS: its shifted label is then
                 closer than any snap (strings 0.5-30 ms apart, partials
                 shared between chord tones)
Neither "none" nor "kept" moves the note beyond the track's shift.

Search windows come from what is known about the labels, in two steps:
    estimate_prior  [label - BEFORE_MS, label + AFTER_MS] for every note;
                    the "precise" snaps of notes alone in their cluster give
                    the track's label shift (median) and jitter (2 x the 75th
                    percentile of |deviation|, clipped to JITTER_MS), when
                    there are at least MIN_ANCHORS of them.
    snap_notes      label + shift +- jitter, for every note.
Windows always keep GAP_MS from the neighbouring labels on the same string.

The thresholds were set on Karplus-Strong scenes with known attacks
(isolated, over ringing strings, re-strikes, strums, soft plucks) and
checked on GuitarSet, GAPS, GOAT and Guitar-TECHS.
"""
from typing import NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

BEFORE_MS, AFTER_MS, GAP_MS = 80, 40, 30
CLUSTER_MS, MATCH_MS = 60, 4
COMB_MIN_DB, BB_MIN_DB, BB_ALONE_DB = 6.0, 3.0, 6.0
CLUSTER_SNAP_MS = 10
HARMONICS = 8
FLOOR_DB = 20
JITTER_MS = (4.0, 30.0)
MIN_ANCHORS = 20

SNAPPED = ("precise", "comb", "transient")


class Note(NamedTuple):
    string: int
    frame: int          # labelled onset frame
    f0: float           # labelled pitch in Hz, 0 if unknown


class Prior(NamedTuple):
    shift_ms: float     # attack minus label centre, median over anchors
    jitter_ms: float    # half-width of the search window around label + shift
    anchors: int        # lone notes the estimate came from (0: given, not measured)


class Snap(NamedTuple):
    sample: int | None  # attack sample, None if not snapped
    status: str         # one of SNAPPED, "none", "kept"
    strength: float     # comb rise in dB


def _samples(milliseconds: float, sample_rate: int) -> int:
    return int(round(milliseconds / 1000 * sample_rate))


def transient_curve(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Broadband rise in dB per step, and the sample at which each rise happens."""
    block = max(32, _samples(128 / 44.1, sample_rate))
    step = block // 4
    highpassed = np.diff(audio, prepend=audio[:1]).astype(np.float64)
    blocks = sliding_window_view(highpassed, block)[::step]
    energy = 10 * np.log10((blocks ** 2).sum(1) + 1e-12)
    rise = np.diff(energy, prepend=energy[:1])
    # Block k adds the samples [k * step + block - step, k * step + block):
    # its rise happens in that new part, timed at its middle.
    times = step * np.arange(len(energy)) + block - step // 2
    return rise, times


def _transient_peaks(rise, times, lo, hi):
    # One step of margin each side, so a peak on the window's edge still has
    # both neighbours to be compared with.
    first = max(int(np.searchsorted(times, lo)) - 1, 0)
    last = int(np.searchsorted(times, hi)) + 1
    peaks, _ = find_peaks(rise[first:last], height=BB_MIN_DB)
    t = times[first + peaks]
    return t[(t >= lo) & (t <= hi)]


def comb_attack(
    audio: np.ndarray, sample_rate: int, f0: float, lo: int, hi: int,
    others: list[float] = (),
) -> tuple[int | None, float]:
    """The attack of a note at f0 within samples [lo, hi]: (sample or None, strength dB).

    `others` are the f0s of notes attacked nearby on other strings, whose
    partials are excluded from the comb.
    """
    window = int(np.clip(round(4 * sample_rate / f0),
                         _samples(10, sample_rate), _samples(46, sample_rate)))
    step = max(8, _samples(32 / 44.1, sample_rate))
    first_end, last_end = lo - window, hi + 2 * window
    if hi < lo or first_end - window < 0 or last_end > len(audio):
        return None, 0.0

    harmonics = np.arange(1, HARMONICS + 1)
    harmonics = harmonics[harmonics * f0 < 0.45 * sample_rate]
    if len(others):
        tolerance = sample_rate / window        # half the Hann main-lobe half-width
        partials = np.concatenate([o * np.arange(1, 4 * HARMONICS + 1) for o in others])
        clash = (np.abs(harmonics[:, None] * f0 - partials[None, :]) < tolerance).any(1)
        if (~clash).any():
            harmonics = harmonics[~clash]

    n = np.arange(window)
    basis = np.hanning(window)[None, :] * np.exp(
        -2j * np.pi * (harmonics[:, None] * f0) * n[None, :] / sample_rate)
    segment = np.asarray(audio[first_end - window:last_end], dtype=np.float64)
    frames = sliding_window_view(segment, window)[::step]   # frame i ends at first_end + i * step
    energy = np.abs(frames @ basis.T) ** 2                   # (ends, harmonics)
    ends = first_end + step * np.arange(len(energy))

    floor = energy.max(0, keepdims=True) * 10 ** (-FLOOR_DB / 10) + 1e-12
    level = 10 * np.log10(energy + floor)
    lag = max(1, int(round(window / step)))
    rise = (level[lag:] - level[:-lag]).mean(1)              # end e against end e - W
    lagged_ends = ends[lag:]
    allowed = (lagged_ends >= lo + window) & (lagged_ends <= hi + window)
    if not allowed.any():
        return None, 0.0
    best = np.flatnonzero(allowed)[int(np.argmax(rise[allowed]))]
    strength = float(rise[best])
    if strength < COMB_MIN_DB:
        return None, strength

    starts_at_attack = lagged_ends[best]                     # this window starts at the attack
    total = energy.sum(1)
    j0 = int(np.searchsorted(ends, starts_at_attack - window))   # ... this one ends there
    j1 = int(np.searchsorted(ends, starts_at_attack + window // 2))
    before, after = total[j0], total[j0:j1 + 1].max()
    crossing = j0 + int(np.argmax(total[j0:j1 + 1] >= 0.5 * (before + after)))
    return int(ends[crossing] - window // 2), strength


def _clusters(label: np.ndarray, sample_rate: int) -> list[list[int]]:
    gap = _samples(CLUSTER_MS, sample_rate)
    clusters, current = [], []
    for i in np.argsort(label, kind="stable"):
        if current and label[i] - label[current[-1]] > gap:
            clusters.append(current)
            current = []
        current.append(int(i))
    if current:
        clusters.append(current)
    return clusters


def _snap_pass(audio, sample_rate, notes, label, centre, half_width, curve, snap_clusters):
    rise, times = curve
    strings = np.array([note.string for note in notes])
    gap = _samples(GAP_MS, sample_rate)
    bounds = []
    for i, string in enumerate(strings):
        lo, hi = centre[i] - half_width[0], centre[i] + half_width[1]
        same = label[(strings == string) & (np.arange(len(notes)) != i)]
        if (same < label[i]).any():
            lo = max(lo, same[same < label[i]].max() + gap)
        if (same > label[i]).any():
            hi = min(hi, same[same > label[i]].min() - gap)
        bounds.append((int(lo), int(hi)))

    near = _samples(BEFORE_MS + AFTER_MS, sample_rate)
    estimates = []
    for i, note in enumerate(notes):
        if note.f0 <= 0:
            estimates.append((None, 0.0))
            continue
        others = [notes[j].f0 for j in np.flatnonzero(np.abs(label - label[i]) <= near)
                  if j != i and notes[j].string != note.string and notes[j].f0 > 0]
        estimates.append(comb_attack(audio, sample_rate, note.f0, *bounds[i], others))

    snaps = [Snap(sample, "comb" if sample is not None else "none", strength)
             for sample, strength in estimates]
    radius = _samples(MATCH_MS, sample_rate)
    clusters = _clusters(label, sample_rate)
    for cluster in clusters:
        if len(cluster) > 1 and not snap_clusters:
            for i in cluster:
                snaps[i] = Snap(None, "kept", estimates[i][1])
            continue
        if len(cluster) == 1 and estimates[cluster[0]][0] is None:
            i = cluster[0]
            a, b = np.searchsorted(times, bounds[i][0]), np.searchsorted(times, bounds[i][1])
            if b > a and rise[a:b].max() >= BB_ALONE_DB:
                snaps[i] = Snap(int(times[a + int(np.argmax(rise[a:b]))]), "transient",
                                estimates[i][1])
            continue
        with_comb = [i for i in cluster if estimates[i][0] is not None]
        if not with_comb:
            continue
        peaks = _transient_peaks(rise, times,
                                 min(bounds[i][0] for i in cluster) - radius,
                                 max(bounds[i][1] for i in cluster) + radius)
        if len(peaks) == 0:
            continue
        combs = np.array([estimates[i][0] for i in with_comb], dtype=np.float64)
        cost = np.abs(combs[:, None] - peaks[None, :])
        rows, cols = linear_sum_assignment(np.where(cost <= radius, cost, 1e9))
        for r, k in zip(rows, cols):
            if cost[r, k] <= radius:
                i = with_comb[r]
                snaps[i] = Snap(int(peaks[k]), "precise", estimates[i][1])
    return snaps, clusters


def _label_centres(notes, hop_size):
    return np.array([(note.frame + 0.5) * hop_size for note in notes])


def estimate_prior(
    audio: np.ndarray, sample_rate: int, hop_size: int, notes: list[Note],
) -> Prior | None:
    """The track's label shift and jitter, from its unambiguous notes.

    None when fewer than MIN_ANCHORS notes give a precise snap on their own:
    the caller should then use its dataset's prior.
    """
    if not notes:
        return None
    audio = np.asarray(audio)
    label = _label_centres(notes, hop_size)
    window = (_samples(BEFORE_MS, sample_rate), _samples(AFTER_MS, sample_rate))
    snaps, clusters = _snap_pass(audio, sample_rate, notes, label, label, window,
                                 transient_curve(audio, sample_rate), snap_clusters=True)
    deviations = np.array([(snaps[c[0]].sample - label[c[0]]) / sample_rate * 1000
                           for c in clusters
                           if len(c) == 1 and snaps[c[0]].status == "precise"])
    if len(deviations) < MIN_ANCHORS:
        return None
    shift = float(np.median(deviations))
    jitter = float(np.clip(2 * np.percentile(np.abs(deviations - shift), 75), *JITTER_MS))
    return Prior(shift, jitter, len(deviations))


def snap_notes(
    audio: np.ndarray, sample_rate: int, hop_size: int, notes: list[Note], prior: Prior,
) -> list[Snap]:
    """Each note's attack, searched within label + prior.shift_ms +- prior.jitter_ms."""
    if not notes:
        return []
    audio = np.asarray(audio)
    label = _label_centres(notes, hop_size)
    half = _samples(prior.jitter_ms, sample_rate)
    snaps, _ = _snap_pass(audio, sample_rate, notes, label,
                          label + _samples(prior.shift_ms, sample_rate), (half, half),
                          transient_curve(audio, sample_rate),
                          snap_clusters=prior.jitter_ms > CLUSTER_SNAP_MS)
    return snaps
