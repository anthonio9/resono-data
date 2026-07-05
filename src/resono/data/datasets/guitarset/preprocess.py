from pathlib import Path

import jams
import librosa
import numpy as np
import soundfile as sf

# GuitarSet native analysis parameters
_NATIVE_HOP   = 256
_NATIVE_SR    = 44100
_NATIVE_HOP_S = _NATIVE_HOP / _NATIVE_SR   # ≈ 5.8 ms


def preprocess(
    raw_dir: Path,
    cache_dir: Path,
    sample_rate: int = 22050,
    hop_size: int = 256,
    remove_overhangs: bool = False,
    overhang_divider: int = 5,
    overhang_threshold_cents: float = 15.0,
) -> None:
    """Convert raw GuitarSet files to .npy cache.

    Produces per-track files in cache_dir/guitarset/:
        {stem}-audio.npy   float32  (N_samples,)
        {stem}-pitch.npy   float32  (6, N_frames)   Hz, 0 = unvoiced
        {stem}-voiced.npy  bool     (6, N_frames)

    Parameters
    ----------
    remove_overhangs:
        If True, silence the tail of each note where the pitch drifts
        significantly — a common artefact on guitar as the string decays.
        The last 1/overhang_divider frames of each note are checked; any
        frame deviating more than overhang_threshold_cents from the note
        body average is marked unvoiced.
    overhang_divider:
        Fraction of the note to treat as the potential overhang tail.
        Default 5 means the last 20% of each note is inspected.
    overhang_threshold_cents:
        Maximum pitch deviation (cents) allowed in the tail before a frame
        is considered an overhang and silenced. Default 15 cents.
    """
    audio_dir = Path(raw_dir) / "guitarset" / "audio_mono-mic"
    ann_dir   = Path(raw_dir) / "guitarset" / "annotation"
    out_dir   = Path(cache_dir) / "guitarset"
    out_dir.mkdir(parents=True, exist_ok=True)

    hop_s = hop_size / sample_rate

    audio_files = sorted(audio_dir.glob("*_mic.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No *_mic.wav files found in {audio_dir}")

    for audio_file in audio_files:
        stem = audio_file.stem.replace("_mic", "")
        jams_file = ann_dir / f"{stem}.jams"
        if not jams_file.exists():
            print(f"  warning: no JAMS for {stem}, skipping")
            continue

        # --- audio ---
        audio, sr = sf.read(audio_file)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        audio = audio.astype(np.float32)
        np.save(out_dir / f"{stem}-audio.npy", audio)

        # --- pitch & voiced ---
        jam = jams.load(str(jams_file))
        n_frames = len(audio) // hop_size
        pitch, voiced = extract_pitch_array_jams(jam, hop_s, n_frames)
        if remove_overhangs:
            pitch, voiced = _remove_pitch_overhangs(
                jam, pitch, voiced, hop_s, n_frames,
                overhang_divider, overhang_threshold_cents,
            )
        np.save(out_dir / f"{stem}-pitch.npy",  pitch)
        np.save(out_dir / f"{stem}-voiced.npy", voiced)

    print(f"Preprocessed {len(audio_files)} tracks → {out_dir}")


# ---------------------------------------------------------------------------
# JAMS helpers
# ---------------------------------------------------------------------------

def extract_pitch_array_jams(
    jam: jams.JAMS,
    hop_size_seconds: float,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-string pitch arrays from a GuitarSet JAMS object.

    Parameters
    ----------
    jam:
        Loaded JAMS object.
    hop_size_seconds:
        Seconds between consecutive pitch frames.
    n_frames:
        Target number of frames (= audio_samples // hop_size).

    Returns
    -------
    pitch  : float32 (6, n_frames)  Hz, 0.0 for unvoiced frames
    voiced : bool    (6, n_frames)
    """
    pitch_anns = jam.annotations["pitch_contour"]
    n_strings  = len(pitch_anns)

    pitch  = np.zeros((n_strings, n_frames), dtype=np.float32)
    voiced = np.zeros((n_strings, n_frames), dtype=bool)

    for s, ann in enumerate(pitch_anns):
        for obs in ann:
            if not obs.value["voiced"] or obs.value["frequency"] == 0:
                continue
            t = obs.time
            t_seconds = t.total_seconds() if hasattr(t, "total_seconds") else float(t)
            frame = int(round(t_seconds / hop_size_seconds))
            if 0 <= frame < n_frames:
                pitch[s, frame]  = obs.value["frequency"]
                voiced[s, frame] = True

    return pitch, voiced


# ---------------------------------------------------------------------------
# Overhang removal
# ---------------------------------------------------------------------------

def _remove_pitch_overhangs(
    jam: jams.JAMS,
    pitch: np.ndarray,
    voiced: np.ndarray,
    hop_size_seconds: float,
    n_frames: int,
    divider: int,
    threshold_cents: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Silence frames at the end of notes where pitch drifts beyond threshold.

    Guitar strings decay after plucking, and the pitch often wanders in the
    final moments of a note before falling silent. These 'overhang' frames
    are technically voiced but carry unreliable pitch information. Removing
    them gives the model cleaner targets.

    The last 1/divider frames of each note are inspected. Any frame whose
    pitch deviates more than threshold_cents from the note body mean is
    silenced (pitch set to 0, voiced set to False). Frames within threshold
    are kept even if they are in the tail.
    """
    pitch_anns = jam.annotations["pitch_contour"]
    pitch  = pitch.copy()
    voiced = voiced.copy()

    for s, ann in enumerate(pitch_anns):
        # Collect per-note observations: note_idx → [(frame, freq), ...]
        notes: dict[int, list[tuple[int, float]]] = {}
        for obs in ann:
            if not obs.value["voiced"] or obs.value["frequency"] == 0:
                continue
            t = obs.time
            t_s = t.total_seconds() if hasattr(t, "total_seconds") else float(t)
            frame = int(round(t_s / hop_size_seconds))
            if not (0 <= frame < n_frames):
                continue
            note_idx = obs.value.get("index", 0)
            notes.setdefault(note_idx, []).append((frame, obs.value["frequency"]))

        for frames_freqs in notes.values():
            frames_freqs.sort(key=lambda x: x[0])
            n = len(frames_freqs)
            if n < divider:
                continue

            tail_start  = n - n // divider
            body_mean   = np.mean([f for _, f in frames_freqs[:tail_start]])

            for frame, freq in frames_freqs[tail_start:]:
                deviation = abs(1200.0 * np.log2(freq / body_mean))
                if deviation > threshold_cents:
                    pitch[s, frame]  = 0.0
                    voiced[s, frame] = False

    return pitch, voiced
