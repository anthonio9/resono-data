import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from resono.data.datasets.gaps import midi as midi_reader
from resono.data.datasets.gaps import score as score_reader
from resono.data.datasets.gaps.align import align, assign_strings, predict_times
from resono.data.datasets.gaps.download import GAPS_DIRNAME, read_track_ids

MANIFEST_NAME = "manifest.json"


class MeasureMismatch(ValueError):
    """The score and the syncpoints disagree on how many measures were played."""


def preprocess(
    raw_dir: Path,
    cache_dir: Path,
    sample_rate: int = 22050,
    hop_size: int = 256,
    tolerance: float = 2.0,
    keep_measure_mismatch: bool = False,
    progress: bool = True,
) -> None:
    """Convert raw GAPS files to .npy cache.

    Produces per-track files in cache_dir/gaps/:
        {stem}-audio.npy   float32  (N_samples,)
        {stem}-pitch.npy   float32  (6, N_frames)   Hz, 0 = unvoiced
        {stem}-voiced.npy  bool     (6, N_frames)
        {stem}-onset.npy   bool     (6, N_frames)   True on note-start frames

    The first three match the GuitarSet cache exactly, so the loader reads
    both datasets unchanged. The onset file is new: unlike voicing transitions
    it marks a re-struck note of the same pitch on the same string, which the
    loader currently cannot recover.

    Labels come from the score rather than from audio analysis, so pitch is
    piecewise constant — GAPS carries no bends or vibrato. Onsets are the
    fine-aligned MIDI's; string assignment is transferred from the MusicXML
    tablature by :mod:`resono.data.datasets.gaps.align` and is approximate.

    Writes manifest.json alongside the arrays, recording which tracks were
    cached and why each of the rest was not. About a quarter of GAPS is
    excluded by default — see `keep_measure_mismatch` — so the record of what
    was dropped is needed to investigate it later.

    Parameters
    ----------
    tolerance:
        Seconds of syncpoint-predicted timing error tolerated when matching a
        MIDI note to a score note. Larger values recover more notes on pieces
        with sparse syncpoints, at the cost of admitting looser matches.
    keep_measure_mismatch:
        Cache tracks whose score and syncpoints disagree on the measure count
        (~28% of GAPS) instead of excluding them. Their labels may be shifted
        against the audio, so this is off by default; turn it on to
        investigate those tracks rather than to train on them.
    progress:
        Show a progress bar over tracks. Enabled by default; pass False (or
        --no-progress-bar on the CLI) to silence it.
    """
    gaps_root = Path(raw_dir) / GAPS_DIRNAME
    out_dir = Path(cache_dir) / GAPS_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    track_ids = read_track_ids(gaps_root)
    if not track_ids:
        raise FileNotFoundError(
            f"No GAPS tracks found under {gaps_root}; run download first"
        )

    hop_seconds = hop_size / sample_rate
    included: list[str] = []
    excluded: dict[str, str] = {}

    for track_id in tqdm(
        track_ids, desc="Preprocessing", unit="track", disable=not progress
    ):
        try:
            arrays = _preprocess_track(
                gaps_root, track_id, sample_rate, hop_size, hop_seconds, tolerance,
                keep_measure_mismatch,
            )
        except MeasureMismatch as error:
            excluded[track_id] = f"measure mismatch: {error}"
            continue
        except (FileNotFoundError, ValueError) as error:
            excluded[track_id] = str(error)
            tqdm.write(f"  warning: skipping {track_id}: {error}")
            continue

        audio, pitch, voiced, onset = arrays
        np.save(out_dir / f"{track_id}-audio.npy", audio)
        np.save(out_dir / f"{track_id}-pitch.npy", pitch)
        np.save(out_dir / f"{track_id}-voiced.npy", voiced)
        np.save(out_dir / f"{track_id}-onset.npy", onset)
        included.append(track_id)

    manifest = out_dir / MANIFEST_NAME
    with open(manifest, "w") as handle:
        json.dump({"included": included, "excluded": excluded}, handle, indent=2)

    mismatched = sum(1 for r in excluded.values() if r.startswith("measure mismatch"))
    print(
        f"Preprocessed {len(included)}/{len(track_ids)} tracks → {out_dir}\n"
        f"  excluded {len(excluded)} ({mismatched} on measure mismatch); "
        f"see {manifest}"
    )


def _preprocess_track(
    gaps_root: Path,
    track_id: str,
    sample_rate: int,
    hop_size: int,
    hop_seconds: float,
    tolerance: float,
    keep_measure_mismatch: bool,
):
    """Build the cache arrays for one track."""
    audio_path = gaps_root / "audio" / f"{track_id}.wav"
    midi_path = gaps_root / "midi" / f"{track_id}.mid"
    score_path = gaps_root / "musicxml" / f"{track_id}.xml"
    sync_path = gaps_root / "syncpoints" / f"{track_id}.json"

    for path in (audio_path, midi_path, score_path, sync_path):
        if not path.exists():
            raise FileNotFoundError(f"missing {path.name}")

    # Annotations are parsed and validated before the audio is decoded:
    # resampling a 48 kHz stereo file is by far the most expensive step here,
    # and about a quarter of tracks are disqualified on their annotations.
    # --- annotations ---
    midi_notes = midi_reader.read_notes(midi_path)
    score_notes = score_reader.read_score(score_path)
    syncpoints = json.loads(sync_path.read_text())
    if not midi_notes or not score_notes:
        raise ValueError("no notes in MIDI or score")

    agree, performed, expected = _check_measures(score_notes, syncpoints)
    if not agree and not keep_measure_mismatch:
        raise MeasureMismatch(
            f"score has {performed} measures, syncpoints describe {expected}"
        )

    predicted = predict_times(score_notes, syncpoints)
    matches = align(midi_notes, score_notes, predicted, tolerance=tolerance)
    # The score usually declares its tuning; fall back to reading it off the
    # tablature when it does not.
    tuning = score_reader.read_tuning(score_path) or score_reader.infer_tuning(
        score_notes
    )
    strings, _from_score = assign_strings(midi_notes, score_notes, matches, tuning)

    # --- audio: GAPS ships 48 kHz stereo ---
    audio, source_rate = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if source_rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=source_rate, target_sr=sample_rate)
    audio = audio.astype(np.float32)

    n_frames = len(audio) // hop_size
    pitch, voiced, onset = _rasterise(midi_notes, strings, n_frames, hop_seconds)
    return audio, pitch, voiced, onset


def _check_measures(score_notes, syncpoints) -> tuple[bool, int, int]:
    """Compare the score's measure count against the syncpoints'.

    Syncpoint indices are performed measure numbers, so they independently
    state how many measures the performance contains. On roughly 28% of GAPS
    the two disagree, and always with the syncpoints claiming more — the
    score, as encoded, does not account for the performed length. Which side
    is wrong is unresolved: candidates are repeats the performer took but the
    score does not mark, the anacrusis handling the GAPS paper lists as a
    known open issue, and syncpoint indices not meaning performed-measure in
    every file.

    It matters because the syncpoints are what map score positions onto
    seconds. If that mapping is wrong the time prior is wrong, and those
    tracks do in fact align worst. So a mismatch disqualifies the track by
    default rather than shipping labels that may be shifted against audio.

    Returns (agree, performed measures, measures the syncpoints describe).
    """
    performed = max(note.measure for note in score_notes) + 1
    indices = [point[0] for point in syncpoints if len(point) >= 2]
    if not indices:
        return True, performed, performed
    expected = max(indices) + 1
    return abs(performed - expected) <= 1, performed, expected


def _rasterise(
    midi_notes: list[tuple[float, float, int]],
    strings: np.ndarray,
    n_frames: int,
    hop_seconds: float,
):
    """Paint notes onto the (6, n_frames) label grids."""
    pitch = np.zeros((score_reader.N_STRINGS, n_frames), dtype=np.float32)
    voiced = np.zeros((score_reader.N_STRINGS, n_frames), dtype=bool)
    onset = np.zeros((score_reader.N_STRINGS, n_frames), dtype=bool)

    for index, (note_on, note_off, note_pitch) in enumerate(midi_notes):
        string = int(strings[index])
        if string < 0:
            continue

        start = int(round(note_on / hop_seconds))
        end = max(start + 1, int(round(note_off / hop_seconds)))
        start = max(0, min(start, n_frames))
        end = max(0, min(end, n_frames))
        if start >= end:
            continue

        # Later notes win the overlap: on one string the newly struck note is
        # what is actually sounding.
        pitch[string, start:end] = 440.0 * 2.0 ** ((note_pitch - 69) / 12.0)
        voiced[string, start:end] = True
        onset[string, start] = True

    return pitch, voiced, onset
