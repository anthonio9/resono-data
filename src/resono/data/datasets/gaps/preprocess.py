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


def preprocess(
    raw_dir: Path,
    cache_dir: Path,
    sample_rate: int = 22050,
    hop_size: int = 256,
    tolerance: float = 2.0,
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

    Parameters
    ----------
    tolerance:
        Seconds of syncpoint-predicted timing error tolerated when matching a
        MIDI note to a score note. Larger values recover more notes on pieces
        with sparse syncpoints, at the cost of admitting looser matches.
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
    written = 0

    for track_id in tqdm(
        track_ids, desc="Preprocessing", unit="track", disable=not progress
    ):
        try:
            arrays = _preprocess_track(
                gaps_root, track_id, sample_rate, hop_size, hop_seconds, tolerance
            )
        except (FileNotFoundError, ValueError) as error:
            tqdm.write(f"  warning: skipping {track_id}: {error}")
            continue

        audio, pitch, voiced, onset = arrays
        np.save(out_dir / f"{track_id}-audio.npy", audio)
        np.save(out_dir / f"{track_id}-pitch.npy", pitch)
        np.save(out_dir / f"{track_id}-voiced.npy", voiced)
        np.save(out_dir / f"{track_id}-onset.npy", onset)
        written += 1

    print(f"Preprocessed {written}/{len(track_ids)} tracks → {out_dir}")


def _preprocess_track(
    gaps_root: Path,
    track_id: str,
    sample_rate: int,
    hop_size: int,
    hop_seconds: float,
    tolerance: float,
):
    """Build the cache arrays for one track."""
    audio_path = gaps_root / "audio" / f"{track_id}.wav"
    midi_path = gaps_root / "midi" / f"{track_id}.mid"
    score_path = gaps_root / "musicxml" / f"{track_id}.xml"
    sync_path = gaps_root / "syncpoints" / f"{track_id}.json"

    for path in (audio_path, midi_path, score_path, sync_path):
        if not path.exists():
            raise FileNotFoundError(f"missing {path.name}")

    # --- audio: GAPS ships 48 kHz stereo ---
    audio, source_rate = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if source_rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=source_rate, target_sr=sample_rate)
    audio = audio.astype(np.float32)

    # --- annotations ---
    midi_notes = midi_reader.read_notes(midi_path)
    score_notes = score_reader.read_score(score_path)
    syncpoints = json.loads(sync_path.read_text())
    if not midi_notes or not score_notes:
        raise ValueError("no notes in MIDI or score")

    _check_unfolding(score_notes, syncpoints, track_id)

    predicted = predict_times(score_notes, syncpoints)
    matches = align(midi_notes, score_notes, predicted, tolerance=tolerance)
    # The score usually declares its tuning; fall back to reading it off the
    # tablature when it does not.
    tuning = score_reader.read_tuning(score_path) or score_reader.infer_tuning(
        score_notes
    )
    strings, _from_score = assign_strings(midi_notes, score_notes, matches, tuning)

    n_frames = len(audio) // hop_size
    pitch, voiced, onset = _rasterise(midi_notes, strings, n_frames, hop_seconds)
    return audio, pitch, voiced, onset


def _check_unfolding(score_notes, syncpoints, track_id: str) -> None:
    """Warn when repeat unfolding disagrees with the syncpoint measure count.

    Syncpoint indices are performed measure numbers, so they independently
    state how many measures the performance contains. A mismatch means the
    repeat structure was expanded wrongly — the labels would then be shifted
    against the audio, which is worth surfacing rather than silently shipping.
    """
    performed = max(note.measure for note in score_notes) + 1
    indices = [point[0] for point in syncpoints if len(point) >= 2]
    if not indices:
        return
    expected = max(indices) + 1
    if abs(performed - expected) > 1:
        tqdm.write(
            f"  warning: {track_id} unfolds to {performed} measures but the "
            f"syncpoints describe {expected}; repeats may be misread"
        )


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
