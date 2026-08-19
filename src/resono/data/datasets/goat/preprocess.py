"""Convert GOAT takes to the resono .npy cache.

GOAT pairs real direct-input electric guitar with Guitar Pro tablature. The
tablature carries string and fret but is quantised to a notated grid; the MIDI
carries performance timing but no string. Labels are therefore a join: notes
and times from the MIDI, string assignment transferred from the tablature by
the GAPS aligner, which GOAT's own alignment procedure is taken from.

What the labels are and are not
-------------------------------
Pitch is piecewise constant. GOAT has no continuous f0, so bends and vibrato
are absent from the labels even where they are audible in the DI.

Onsets are trustworthy, durations less so. GOAT's alignment moves onsets only
— measured across the takes that aligned, chord notes spread from a mean of
0.1 ms apart to 3.4 ms, while the piece's end time moves by 0.05% — and the
alignment quality score the dataset ships is an onset-only F-measure. Note
ends are inherited from the notated grid and nothing validates them.

Dynamics are absent by construction: Guitar Pro up to version 5 stores no note
velocity, so loudness survives only in the audio.
"""
import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from resono.data.datasets.gaps import midi as midi_reader
from resono.data.datasets.gaps.align import align, assign_strings
from resono.data.datasets.gaps.score import N_STRINGS
from resono.data.datasets.goat import score as score_reader

#: Range searched when reconciling tablature pitch against the MIDI. Wide
#: enough for an octave either way, narrow enough that a genuine mismatch
#: cannot be papered over by finding some shift that happens to fit.
TRANSPOSE_SEARCH = range(-12, 13)

#: How far a shift must beat leaving the pitches alone before it is applied.
#: The two takes that need it jump from near zero to 95%+, while a correctly
#: parsed take sits near 99% at zero and no rival shift comes close.
TRANSPOSE_MARGIN = 0.25

DATASET_NAME = "goat"
GOAT_DIRNAME = "GOAT"
MANIFEST_NAME = "manifest.json"

#: metadata.csv column holding each audio variant.
AUDIO_SOURCES = {
    "di": "di_audio_path",
    "amp1": "amp_audio_path_1", "amp2": "amp_audio_path_2",
    "amp3": "amp_audio_path_3", "amp4": "amp_audio_path_4",
    "amp5": "amp_audio_path_5",
    "gp": "gp_audio_path",
}


def preprocess(
    raw_dir: Path,
    cache_dir: Path,
    sample_rate: int = 22050,
    hop_size: int = 256,
    audio_source: str = "di",
    alignment_threshold: float = 0.5,
    disable_unaligned: bool = False,
    tolerance: float = 2.0,
    dataset_name: str = DATASET_NAME,
    progress: bool = True,
) -> None:
    """Convert raw GOAT files to .npy cache.

    Produces per-take files in cache_dir/goat/:
        {stem}-audio.npy   float32  (N_samples,)
        {stem}-pitch.npy   float32  (6, N_frames)   Hz, 0 = unvoiced
        {stem}-voiced.npy  bool     (6, N_frames)
        {stem}-onset.npy   bool     (6, N_frames)   True on note-start frames

    Parameters
    ----------
    audio_source:
        Which recording to cache. 'di' is the raw pickup signal and the one
        that matches a plugged-in guitar; amp1-5 are reamps of that same DI
        through Neural Amp Modeler profiles, so they carry identical labels
        and are timbre augmentation rather than new material.
    alignment_threshold:
        Minimum ``alignment_f_measure_fine``. That column is an onset-only
        F-measure between two runs of the aligner — a self-consistency check,
        not an accuracy score — and it correlates *negatively* with harmonic
        density (Spearman -0.37 over the 152 scored takes), because the fine
        stage exists to resolve chord asynchrony and chords are where onsets
        are most ambiguous. Gating at 0.7 would discard takes twice as dense
        as those it keeps. The default of 0.5 removes only the seven takes
        where the aligner failed outright, which are not dense.
    disable_unaligned:
        Drop the 20 takes that ship no fine-aligned MIDI. They are kept by
        default: their quantised MIDI still resolves onsets to a median of
        0.9 ms where alignment did run, and excluding them costs a third of
        one player's material and a third of the Strandberg takes in a corpus
        already 77% one player and 63% one guitar. The GOAT paper states all
        172 takes carry fine-aligned MIDI; 20 do not.
    tolerance:
        Seconds of notated-timing error tolerated when matching a MIDI note to
        a tablature note.
    """
    if audio_source not in AUDIO_SOURCES:
        raise ValueError(
            f"audio_source must be one of {sorted(AUDIO_SOURCES)}, got {audio_source!r}"
        )

    goat_root = Path(raw_dir) / GOAT_DIRNAME
    metadata_path = goat_root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found. GOAT is distributed by request via Zenodo "
            "record 15690894; extract it so that metadata.csv sits at this path."
        )
    rows = list(csv.DictReader(metadata_path.open()))

    out_dir = Path(cache_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    hop_seconds = hop_size / sample_rate
    cached: list[str] = []
    skipped: dict[str, str] = {}
    transposed: dict[str, int] = {}

    for row in tqdm(rows, desc="Preprocessing", unit="take", disable=not progress):
        stem = row["item"]
        try:
            arrays = _preprocess_take(
                goat_root, row, sample_rate, hop_size, hop_seconds,
                audio_source, alignment_threshold, disable_unaligned, tolerance,
            )
        except Exception as error:
            skipped[stem] = f"{type(error).__name__}: {error}"
            continue
        audio, pitch, voiced, onset, shift = arrays
        if shift:
            transposed[stem] = shift
        np.save(out_dir / f"{stem}-audio.npy", audio)
        np.save(out_dir / f"{stem}-pitch.npy", pitch)
        np.save(out_dir / f"{stem}-voiced.npy", voiced)
        np.save(out_dir / f"{stem}-onset.npy", onset)
        cached.append(stem)

    manifest = {
        "audio_source": audio_source,
        "alignment_threshold": alignment_threshold,
        "disable_unaligned": disable_unaligned,
        "sample_rate": sample_rate,
        "hop_size": hop_size,
        "cached": sorted(cached),
        "skipped": skipped,
        # Recorded so a silent correction cannot hide a real parsing bug.
        "transposed": transposed,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    print(f"Cached {len(cached)} of {len(rows)} takes ({audio_source}) → {out_dir}")
    if skipped:
        print(f"  skipped {len(skipped)}; see {MANIFEST_NAME}")
    if transposed:
        print(f"  reconciled a pitch offset on {len(transposed)}: {transposed}")


def _resolve(goat_root: Path, value: str) -> Path | None:
    """metadata.csv paths are written relative to the GOAT/ directory."""
    value = value.strip()
    if not value:
        return None
    relative = value[len(GOAT_DIRNAME) + 1:] if value.startswith(GOAT_DIRNAME + "/") else value
    path = goat_root / "data" / relative
    return path if path.exists() else None


def _preprocess_take(
    goat_root, row, sample_rate, hop_size, hop_seconds,
    audio_source, alignment_threshold, disable_unaligned, tolerance,
):
    """Build the cache arrays for one take."""
    score_measure = row.get("alignment_f_measure_fine", "").strip()
    fine = _resolve(goat_root, row.get("finealigned_midi_path", ""))
    if fine is None:
        if disable_unaligned:
            raise ValueError("no fine-aligned MIDI")
        midi_path = _resolve(goat_root, row.get("unaligned_midi_path", ""))
        if midi_path is None:
            raise FileNotFoundError("no MIDI of either kind")
    else:
        midi_path = fine
        if score_measure and float(score_measure) < alignment_threshold:
            raise ValueError(f"alignment F {float(score_measure):.3f} below threshold")

    tab_path = _resolve(goat_root, row.get("gp5_path", ""))
    if tab_path is None:
        raise FileNotFoundError("no Guitar Pro tablature")
    audio_path = _resolve(goat_root, row.get(AUDIO_SOURCES[audio_source], ""))
    if audio_path is None:
        raise FileNotFoundError(f"no {audio_source} audio")

    # --- annotations first: decoding audio is the expensive step ---
    midi_notes = midi_reader.read_notes(midi_path)
    tablature = score_reader.read_score(tab_path)
    score_notes, ticks = tablature.notes, tablature.ticks
    if not midi_notes or not score_notes:
        raise ValueError("no notes in MIDI or tablature")

    # Two takes (item_54, item_95) have tablature written an octave below the
    # MIDI, and nothing in the .gp5 records it — same instrument, tuning and
    # capo as takes that agree. Rather than trust either side, find the
    # constant shift that best reconciles the two pitch sets.
    shift = _reconcile_transposition(score_notes, midi_notes)
    if shift:
        score_notes = [replace(note, pitch=note.pitch + shift)
                       for note in score_notes]

    predicted = np.asarray(score_reader.predict_times(ticks, tablature.tempos))
    matches = align(midi_notes, score_notes, predicted, tolerance=tolerance)
    tuning = tablature.tuning
    strings, _from_score = assign_strings(midi_notes, score_notes, matches, tuning)

    # --- audio: 44.1 kHz, and three of the 172 DI files are stereo ---
    audio, source_rate = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if source_rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=source_rate, target_sr=sample_rate)
    audio = audio.astype(np.float32)

    n_frames = len(audio) // hop_size
    pitch, voiced, onset = _rasterise(midi_notes, strings, n_frames, hop_seconds)
    return audio, pitch, voiced, onset, shift


def _reconcile_transposition(score_notes, midi_notes) -> int:
    """Constant semitone shift that best aligns the two pitch histograms.

    Counters rather than sets, so the comparison respects how often each pitch
    is played: their intersection takes min(count) per pitch, which a set
    would flatten to mere presence.

    Returns 0 unless a non-zero shift clearly beats it, so the common case
    costs nothing and an ambiguous one is left alone for the aligner to handle
    — or to fail on visibly.
    """
    score_pitches = Counter(note.pitch for note in score_notes)
    midi_pitches = Counter(pitch for _, _, pitch in midi_notes)
    total = max(sum(score_pitches.values()), 1)

    scores = {
        shift: sum(
            (Counter({p + shift: n for p, n in score_pitches.items()})
             & midi_pitches).values()
        ) / total
        for shift in TRANSPOSE_SEARCH
    }
    best = max(scores, key=scores.get)
    return best if best != 0 and scores[best] > scores[0] + TRANSPOSE_MARGIN else 0


def _rasterise(midi_notes, strings, n_frames, hop_seconds):
    """Paint notes onto the (6, n_frames) label grids."""
    pitch = np.zeros((N_STRINGS, n_frames), dtype=np.float32)
    voiced = np.zeros((N_STRINGS, n_frames), dtype=bool)
    onset = np.zeros((N_STRINGS, n_frames), dtype=bool)

    for index, (note_on, note_off, note_pitch) in enumerate(midi_notes):
        string = int(strings[index])
        if string < 0:
            continue
        start = max(0, min(int(round(note_on / hop_seconds)), n_frames))
        end = max(0, min(max(start + 1, int(round(note_off / hop_seconds))), n_frames))
        if start >= end:
            continue
        # Later notes win the overlap: on one string the newly struck note is
        # what is actually sounding.
        pitch[string, start:end] = 440.0 * 2.0 ** ((note_pitch - 69) / 12.0)
        voiced[string, start:end] = True
        onset[string, start] = True

    return pitch, voiced, onset
