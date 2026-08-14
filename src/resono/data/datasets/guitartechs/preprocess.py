"""Convert Guitar-TECHS to the shared .npy cache.

Produces the same four arrays as guitarset and gaps, so the loader reads all
three without knowing the difference. What differs is where the labels come
from: a Fishman Triple Play pickup writing one MIDI note per plucked string,
rather than a pitch tracker run over audio.

That has one consequence worth stating plainly. MIDI carries **no pitch
contour** — a note is a single number held for its duration, and these files
contain no pitchwheel messages at all. Sustained notes are labelled correctly;
expressive pitch is not recorded anywhere. Four takes are therefore excluded
by default (see :data:`DEFAULT_EXCLUDE`), all sharing one property: the
pickup's note number is not the pitch in the air. Their onsets and string
assignment remain sound, so all four are worth revisiting with an
audio-derived f0 rather than discarding.

What is kept is the material where a note is struck and held at a fixed pitch
— chords, scales, single notes and palm mutes — which is 86% of the dataset
and includes the systematically enumerated chord vocabulary.

Two systematic corrections are applied, both measured rather than assumed:
the ~23 ms pickup latency (see :mod:`guitartechs.midi`) and, optionally, the
string-dependent tuning offset (see ``tuning_offset_cents``).
"""
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from resono.data.datasets.guitartechs.midi import (
    MERGE_GAP_MS,
    N_STRINGS,
    ONSET_LATENCY_MS,
    Note,
    read_notes,
)

DATASET_NAME = "guitartechs"

# The four synchronised recordings of every take. 'directinput' is the raw
# pickup signal — no amp, no cabinet, no room, no microphone — which is the
# most consistent across the dataset's deliberately varied hardware and the
# easiest for a pitch model to read. The others exist for tone augmentation
# and for training robustness to a signal chain.
AUDIO_SOURCES = {
    "directinput": ("audio/directinput", "directinput", ".wav"),
    "micamp":      ("audio/micamp",      "micamp",      ".wav"),
    "ego":         ("video/ego",         "ego",         ".mp3"),
    "exo":         ("video/exo",         "exo",         ".mp3"),
}

# Takes whose labels fail in a way no parsing can repair. All four are
# excluded for their *pitch and coverage* only — onsets, offsets and string
# assignment remain sound in each, so all four are worth revisiting once
# pitch can be estimated from the audio instead.
#
# Bendings: the pickup renders a bend as a staircase of semitone steps, so the
# gesture is tracked but quantised to +/-50 cents, and P1's take additionally
# suffers seconds-long dropouts mid-bend plus occasional octave glitches that
# merging absorbs into a confidently wrong label.
#
# Vibrato: the labels contain no vibrato at all. The file holds only note_on,
# note_off and metadata — no pitchwheel, no control_change, nothing — so each
# note is a single number held flat through an excursion measured at +/-12
# cents around 4.1 Hz. As supervision it is indistinguishable from SingleNotes.
#
# Harmonics and PinchHarmonics: the pickup reports the note it believes was
# *fretted*, but a harmonic sounds at a multiple of the open string, so the
# label names a different pitch from the one in the air — observed an octave
# low, and jumping mid-note between the two. Many events are also so short
# they rasterise to an onset with no voiced span, and P1's PinchHarmonics take
# carries only 99 MIDI notes against the 132 the authors record as played, a
# quarter of its audible notes having no label at all. No merge rule recovers
# a note that was never written, and none corrects a pitch that names the
# wrong thing.
#
# Together ~43 minutes of the ~5h12m dataset, so this costs about 14%.
DEFAULT_EXCLUDE = ("Bendings", "Vibrato", "PinchHarmonics", "Harmonics")


def preprocess(
    raw_dir: Path,
    cache_dir: Path,
    sample_rate: int = 22050,
    hop_size: int = 256,
    audio_source: str = "directinput",
    onset_latency_ms: float = ONSET_LATENCY_MS,
    merge_gap_ms: float = MERGE_GAP_MS,
    tuning_offset_cents: float = 0.0,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
    progress: bool = True,
) -> None:
    """Convert raw Guitar-TECHS files to .npy cache.

    Produces per-take files in cache_dir/guitartechs/:
        {stem}-audio.npy   float32  (N_samples,)
        {stem}-pitch.npy   float32  (6, N_frames)   Hz, 0 = unvoiced
        {stem}-voiced.npy  bool     (6, N_frames)
        {stem}-onset.npy   bool     (6, N_frames)   True on note-start frames

    Stems are ``{session}_{take}``, e.g. ``P1_techniques_Vibrato``.

    Parameters
    ----------
    audio_source:
        Which of the four synchronised recordings to cache. Defaults to the
        direct input; see :data:`AUDIO_SOURCES`.
    onset_latency_ms:
        Subtracted from every note time to undo the pickup's detection delay.
    merge_gap_ms:
        A note starting within this of the previous one on the same string is
        the same pluck, its pitch having crossed a semitone boundary, so it
        gets no onset of its own. Pitch and voicing still follow each MIDI
        note, so a bend remains a rising staircase.
    tuning_offset_cents:
        Added to every labelled pitch. The instruments are not tuned to A440:
        measured on P1, the sounding pitch sits a median 12 cents sharp of the
        MIDI note number, varying by string from -2 to +19 cents. Left at 0 by
        default because the correction is per-string and per-session, so a
        single global number would be wrong for most of it — and 12 cents is
        well inside a 50-cent evaluation threshold.
    exclude:
        Take names to skip entirely. See :data:`DEFAULT_EXCLUDE`.
    """
    if audio_source not in AUDIO_SOURCES:
        raise ValueError(
            f"audio_source must be one of {sorted(AUDIO_SOURCES)}, got {audio_source!r}"
        )

    root = Path(raw_dir) / "guitar-techs"
    out_dir = Path(cache_dir) / DATASET_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    takes = _discover(root, audio_source, exclude)
    if not takes:
        raise FileNotFoundError(
            f"No Guitar-TECHS takes found under {root} for audio source "
            f"{audio_source!r}. Has 'guitartechs download' been run?"
        )

    hop_seconds = hop_size / sample_rate
    written = 0
    for stem, audio_path, midi_path in tqdm(
        takes, desc="Preprocessing", unit="take", disable=not progress
    ):
        audio = _load_audio(audio_path, sample_rate)
        n_frames = len(audio) // hop_size
        if n_frames == 0:
            tqdm.write(f"  warning: {stem} is shorter than one frame, skipping")
            continue

        notes = read_notes(
            midi_path, onset_latency_ms=onset_latency_ms, merge_gap_ms=merge_gap_ms
        )
        pitch, voiced, onset = to_arrays(
            notes, n_frames, hop_seconds, tuning_offset_cents
        )

        np.save(out_dir / f"{stem}-audio.npy", audio)
        np.save(out_dir / f"{stem}-pitch.npy", pitch)
        np.save(out_dir / f"{stem}-voiced.npy", voiced)
        np.save(out_dir / f"{stem}-onset.npy", onset)
        written += 1

    print(f"Preprocessed {written} takes ({audio_source}) → {out_dir}")


def to_arrays(
    notes: list[Note],
    n_frames: int,
    hop_seconds: float,
    tuning_offset_cents: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterise note events onto the frame grid.

    Frame f covers time f * hop_seconds, matching the loader's convention that
    frame f starts at sample f * hop_size.

    Onsets are marked from each note's start rather than derived from voicing
    transitions. A re-struck note at the same pitch leaves no voicing gap, so a
    0->1 derivation misses it — about 22% of onsets in GuitarSet. MIDI gives
    the note boundary directly, at 1.04 ms precision, so no derivation is
    needed here at all.
    """
    pitch = np.zeros((N_STRINGS, n_frames), dtype=np.float32)
    voiced = np.zeros((N_STRINGS, n_frames), dtype=bool)
    onset = np.zeros((N_STRINGS, n_frames), dtype=bool)

    scale = 2.0 ** (tuning_offset_cents / 1200.0)

    for note in notes:
        first = int(round(note.start / hop_seconds))
        last = int(round(note.end / hop_seconds))
        if last <= first:
            last = first + 1
        first = max(first, 0)
        last = min(last, n_frames)
        if first >= n_frames or last <= first:
            continue

        hz = float(librosa.midi_to_hz(note.midi)) * scale
        pitch[note.string, first:last] = np.float32(hz)
        voiced[note.string, first:last] = True
        # Only a real pluck marks an onset; a semitone-crossing continuation
        # is the same note and must not look like a new attack.
        if not note.continues:
            onset[note.string, first] = True

    return pitch, voiced, onset


def _discover(
    root: Path, audio_source: str, exclude: tuple[str, ...]
) -> list[tuple[str, Path, Path]]:
    """Find (stem, audio, midi) triples for every take that has both."""
    subdir, prefix, suffix = AUDIO_SOURCES[audio_source]
    takes = []
    for session in sorted(p for p in root.glob("P*") if p.is_dir()):
        for audio_path in sorted((session / subdir).glob(f"{prefix}_*{suffix}")):
            take = audio_path.stem[len(prefix) + 1:]
            if take in exclude:
                continue
            midi_path = session / "midi" / f"midi_{take}.mid"
            if not midi_path.exists():
                continue
            takes.append((f"{session.name}_{take}", audio_path, midi_path))
    return takes


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Read a take as mono float32 at the target rate.

    Rate and channel count are read per file rather than assumed: the dataset
    mixes 44.1 and 48 kHz and mono and stereo, sometimes within one session.
    """
    audio, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if source_rate != sample_rate:
        audio = librosa.resample(
            audio, orig_sr=source_rate, target_sr=sample_rate, res_type="soxr_hq"
        )
    return audio.astype(np.float32)
