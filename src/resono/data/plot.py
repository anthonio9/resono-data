"""Plot preprocessed labels on top of the audio they describe.

Reads the .npy cache, so it works for any dataset in the registry — the cache
format is shared. Its purpose is to answer one question by eye: do the labels
land where the notes actually are?

The pitch contour is drawn over a constant-Q spectrogram because CQT bins are
logarithmic in frequency, matching how pitch labels are spaced; a correct label
sits on the fundamental with harmonics stacked evenly above it. A linear-frequency
STFT would crowd the low strings into a few pixels and hide exactly the errors
worth seeing.

Strings are separated by position rather than colour. Six categorical hues
cannot be told apart reliably once they overlap — the palette check fails at
six slots on colour-vision-deficiency separation — so the per-string panel
gives each string its own labelled lane instead.
"""
from pathlib import Path

import numpy as np

# Categorical slots 1 and 6, validated for both surfaces at all pairs.
_INK = {
    "light": {
        "surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
        "muted": "#8a8880", "pitch": "#2a78d6", "onset": "#eb6834",
        "spectrogram": "Greys",
    },
    "dark": {
        "surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7",
        "muted": "#75736a", "pitch": "#3987e5", "onset": "#d95926",
        "spectrogram": "gray",
    },
}

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi: float) -> str:
    """Render a MIDI note number as a pitch name, e.g. 64 -> 'E4'."""
    midi = int(round(midi))
    return f"{_NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def load_track(cache_dir: Path, dataset: str, stem: str) -> dict:
    """Load one preprocessed track's arrays.

    ``onset`` is None for datasets that do not write it — GuitarSet derives
    onsets from voicing transitions instead, so the file simply does not exist.
    """
    root = Path(cache_dir) / dataset
    missing = [
        name for name in ("audio", "pitch", "voiced")
        if not (root / f"{stem}-{name}.npy").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{stem} in {root} is missing {', '.join(missing)}; "
            "has it been preprocessed?"
        )

    onset_path = root / f"{stem}-onset.npy"
    return {
        "audio": np.load(root / f"{stem}-audio.npy"),
        "pitch": np.load(root / f"{stem}-pitch.npy"),
        "voiced": np.load(root / f"{stem}-voiced.npy"),
        "onset": np.load(onset_path) if onset_path.exists() else None,
    }


def plot_track(
    cache_dir: Path,
    dataset: str,
    stem: str,
    start: float = 0.0,
    duration: float = 10.0,
    sample_rate: int = 22050,
    hop_size: int = 256,
    theme: str = "light",
    output: Path | None = None,
):
    """Plot labels over audio for one track and return the figure.

    Parameters
    ----------
    start, duration:
        Window in seconds. The default 10 s is about as much as stays legible
        at a normal figure width; beyond roughly 30 s individual notes are
        narrower than a line width and the plot stops being diagnostic.
    theme:
        'light' or 'dark'. Both are selected against their own surface rather
        than one being an inversion of the other.
    """
    import matplotlib.pyplot as plt
    import librosa

    if theme not in _INK:
        raise ValueError(f"theme must be one of {sorted(_INK)}")
    ink = _INK[theme]

    track = load_track(cache_dir, dataset, stem)
    audio, pitch, voiced = track["audio"], track["pitch"], track["voiced"]
    onset = track["onset"]

    # Frame f starts at sample f * hop_size — the loader's convention, and the
    # only thing tying labels to audio. Slicing both from the same frame
    # indices is what keeps the two panels honest.
    hop_seconds = hop_size / sample_rate
    n_frames = pitch.shape[-1]
    first = max(0, int(start / hop_seconds))
    last = min(n_frames, int((start + duration) / hop_seconds))
    if first >= last:
        raise ValueError(
            f"window {start}-{start + duration}s is empty; track is "
            f"{n_frames * hop_seconds:.1f}s long"
        )

    times = (np.arange(first, last) * hop_seconds)
    clip = audio[first * hop_size : last * hop_size]
    pitch_win = np.atleast_2d(pitch)[:, first:last]
    voiced_win = np.atleast_2d(voiced)[:, first:last]
    onset_win = np.atleast_2d(onset)[:, first:last] if onset is not None else None
    n_strings = pitch_win.shape[0]

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(13, 7.5), height_ratios=[3, 2], sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    figure.patch.set_facecolor(ink["surface"])

    _plot_spectrogram(
        top, clip, pitch_win, voiced_win, onset_win, times,
        sample_rate, hop_size, start, ink,
    )
    _plot_string_lanes(bottom, pitch_win, voiced_win, onset_win, times, ink)

    voiced_pct = voiced_win.any(axis=0).mean()
    onsets = "" if onset_win is not None else "  ·  no onset labels in this cache"
    top.set_title(
        f"{dataset} · {stem}   {start:.1f}–{start + duration:.1f}s   "
        f"labelled pitch over CQT   ({voiced_pct:.0%} of frames voiced){onsets}",
        color=ink["primary"], fontsize=12, loc="left", pad=12,
    )
    bottom.set_xlabel("Time (s)", color=ink["secondary"], fontsize=10)
    bottom.set_xlim(times[0], times[-1])

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output, dpi=140, bbox_inches="tight", facecolor=ink["surface"]
        )
        print(f"Saved → {output}")
    return figure


def _plot_spectrogram(
    axes, clip, pitch_win, voiced_win, onset_win, times,
    sample_rate, hop_size, start, ink,
):
    """CQT with the labelled pitch drawn over it."""
    import librosa

    # Span the guitar's range with a little headroom for harmonics: from C1,
    # comfortably below the low E, up over 6 octaves.
    fmin = librosa.note_to_hz("C1")
    bins_per_octave = 36
    n_bins = 6 * bins_per_octave

    cqt = np.abs(librosa.cqt(
        clip, sr=sample_rate, hop_length=hop_size, fmin=fmin,
        n_bins=n_bins, bins_per_octave=bins_per_octave,
    ))
    decibels = librosa.amplitude_to_db(cqt, ref=np.max)

    frames = min(decibels.shape[1], len(times))
    axes.pcolormesh(
        times[:frames],
        librosa.cqt_frequencies(n_bins, fmin=fmin, bins_per_octave=bins_per_octave),
        decibels[:, :frames],
        cmap=ink["spectrogram"], vmin=-60, vmax=0, shading="nearest", rasterized=True,
    )

    # Break the contour at unvoiced frames so rests are not bridged by a line
    # that implies a sounding note.
    for string in range(pitch_win.shape[0]):
        series = np.where(voiced_win[string], pitch_win[string], np.nan)
        axes.plot(
            times, series, color=ink["pitch"], linewidth=2.0,
            solid_capstyle="round", zorder=3,
            label="labelled pitch" if string == 0 else None,
        )

    marked = 0
    if onset_win is not None:
        onset_times, onset_freqs = _onset_points(onset_win, pitch_win, times)
        marked = len(onset_times)
        if marked:
            axes.scatter(
                onset_times, onset_freqs, s=26, color=ink["onset"],
                edgecolors=ink["surface"], linewidths=0.8, zorder=4,
                label="onset",
            )

    axes.set_yscale("log")
    sounding = pitch_win[voiced_win]
    if sounding.size:
        axes.set_ylim(max(fmin, sounding.min() * 0.5), sounding.max() * 4.5)
    axes.set_ylabel("Frequency (Hz)", color=ink["secondary"], fontsize=10)
    _style(axes, ink)

    # One series is named by the title; a legend earns its space only once
    # there are two kinds of mark to tell apart.
    if marked:
        legend = axes.legend(
            loc="upper right", frameon=True, fontsize=9,
            facecolor=ink["surface"], edgecolor=ink["muted"],
        )
        for text in legend.get_texts():
            text.set_color(ink["secondary"])


def _plot_string_lanes(axes, pitch_win, voiced_win, onset_win, times, ink):
    """One labelled lane per string: where it sounds, and at what pitch."""
    n_strings = pitch_win.shape[0]
    step = times[1] - times[0] if len(times) > 1 else 0.01

    for string in range(n_strings):
        lane = n_strings - 1 - string          # index 0 (low E) at the bottom
        active = voiced_win[string]
        if active.any():
            # Constant height: this panel answers "which string, when".
            # Encoding pitch as bar height would not be comparable between
            # lanes, since each string covers a different range — the
            # spectrogram above carries pitch on a real axis instead.
            axes.bar(
                times[active], np.full(active.sum(), 0.52), width=step,
                bottom=lane - 0.26, color=ink["pitch"], edgecolor="none", zorder=2,
            )

        if onset_win is not None and onset_win[string].any():
            axes.vlines(
                times[onset_win[string]], lane - 0.38, lane + 0.38,
                color=ink["onset"], linewidth=1.6, zorder=3,
            )

        axes.axhline(lane - 0.44, color=ink["muted"], linewidth=0.4, alpha=0.35, zorder=1)

    axes.set_yticks(range(n_strings))
    axes.set_yticklabels(
        [_lane_label(pitch_win, voiced_win, n_strings - 1 - lane)
         for lane in range(n_strings)],
        color=ink["secondary"], fontsize=9,
    )
    axes.set_ylim(-0.6, n_strings - 0.4)
    axes.set_ylabel("String  (0 = low E)", color=ink["secondary"], fontsize=10)
    _style(axes, ink)


def _lane_label(pitch_win, voiced_win, string: int) -> str:
    """Lane label: string index and the pitch range actually seen on it.

    Shown as a range rather than a single note so it cannot be misread as the
    open-string tuning — a string's lowest note in some window is usually
    fretted, not open.
    """
    active = voiced_win[string]
    if not active.any():
        return f"{string}   silent"

    span = pitch_win[string][active]
    low = note_name(69 + 12 * np.log2(span.min() / 440.0))
    high = note_name(69 + 12 * np.log2(span.max() / 440.0))
    return f"{string}   {low}" if low == high else f"{string}   {low}–{high}"


def _onset_points(onset_win, pitch_win, times):
    """Onset marks placed at the pitch of the note they start."""
    strings, frames = np.nonzero(onset_win)
    return times[frames], pitch_win[strings, frames]


def _style(axes, ink):
    """Recessive axes: the data should be the only thing with weight."""
    axes.set_facecolor(ink["surface"])
    for side, visible in (("top", False), ("right", False),
                          ("left", True), ("bottom", True)):
        axes.spines[side].set_visible(visible)
        if visible:
            axes.spines[side].set_color(ink["muted"])
            axes.spines[side].set_linewidth(0.8)
    axes.tick_params(colors=ink["secondary"], labelsize=9, width=0.8)
    axes.grid(False)
