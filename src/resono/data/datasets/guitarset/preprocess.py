from pathlib import Path

import jams
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

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
    progress: bool = True,
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
    progress:
        Show a progress bar over tracks. Enabled by default; pass False (or
        --no-progress-bar on the CLI) to silence it.
    """
    gset_root = Path(raw_dir) / "guitarset"
    out_dir   = Path(cache_dir) / "guitarset"
    out_dir.mkdir(parents=True, exist_ok=True)

    hop_s = hop_size / sample_rate

    # Discover files with rglob so the exact extraction layout doesn't matter:
    # mono-mic audio ends in '_mic.wav', annotations in '.jams', wherever they sit.
    audio_files = sorted(gset_root.rglob("*_mic.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No *_mic.wav files found under {gset_root}")
    jams_index = {p.stem: p for p in gset_root.rglob("*.jams")}

    for audio_file in tqdm(
        audio_files, desc="Preprocessing", unit="track", disable=not progress
    ):
        stem = audio_file.stem.replace("_mic", "")
        jams_file = jams_index.get(stem)
        if jams_file is None:
            tqdm.write(f"  warning: no JAMS for {stem}, skipping")
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
        pitch, voiced, note_ids = extract_pitch_note_arrays_jams(jam, hop_s, n_frames)
        if remove_overhangs:
            pitch, voiced = remove_pitch_overhangs(
                pitch, voiced, note_ids,
                overhang_divider, overhang_threshold_cents,
            )
        np.save(out_dir / f"{stem}-pitch.npy",  pitch)
        np.save(out_dir / f"{stem}-voiced.npy", voiced)

    print(f"Preprocessed {len(audio_files)} tracks → {out_dir}")


# ---------------------------------------------------------------------------
# JAMS → uniform grid
# ---------------------------------------------------------------------------

def _obs_seconds(t) -> float:
    """JAMS observation times are pd.Timedelta from disk but float when built
    programmatically — normalise both to seconds."""
    return t.total_seconds() if hasattr(t, "total_seconds") else float(t)


def extract_pitch_array_jams(
    jam: jams.JAMS,
    hop_size_seconds: float,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-string pitch and voicing arrays from a GuitarSet JAMS object.

    Thin wrapper over :func:`extract_pitch_note_arrays_jams` for callers that
    only need the pitch/voicing grids and not the per-frame note ids.

    Returns
    -------
    pitch  : float32 (6, n_frames)  Hz, 0.0 for unvoiced frames
    voiced : bool    (6, n_frames)
    """
    pitch, voiced, _ = extract_pitch_note_arrays_jams(jam, hop_size_seconds, n_frames)
    return pitch, voiced


def extract_pitch_note_arrays_jams(
    jam: jams.JAMS,
    hop_size_seconds: float,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-string pitch, voicing, and note-id arrays from a JAMS object.

    The raw ``pitch_contour`` observations sit on GuitarSet's native ~5.8 ms
    grid. This resamples them onto the target frame grid defined by
    ``hop_size_seconds`` using per-note linear interpolation in log-frequency
    space. Unlike a plain nearest-frame assignment, interpolation fills the
    interior of sustained notes even when the target hop is *finer* than the
    native one, so no spurious unvoiced holes appear inside a held note.

    Parameters
    ----------
    jam:
        Loaded JAMS object.
    hop_size_seconds:
        Seconds between consecutive target frames. Frame f starts at time
        f * hop_size_seconds, matching the loader's "frame f starts at
        sample f * hop_size" convention.
    n_frames:
        Target number of frames (= audio_samples // hop_size).

    Returns
    -------
    pitch    : float32 (6, n_frames)  Hz, 0.0 for unvoiced frames
    voiced   : bool    (6, n_frames)
    note_ids : int32   (6, n_frames)  per-frame note index, -1 where unvoiced.
        Ids are assigned per string in note order (0, 1, 2, …); combined with
        the string axis they identify each note uniquely and let downstream
        steps (e.g. overhang removal) recover exact note boundaries.
    """
    pitch_anns = jam.annotations["pitch_contour"]
    n_strings  = len(pitch_anns)

    pitch    = np.zeros((n_strings, n_frames), dtype=np.float32)
    voiced   = np.zeros((n_strings, n_frames), dtype=bool)
    note_ids = np.full((n_strings, n_frames), -1, dtype=np.int32)

    for s, ann in enumerate(pitch_anns):
        times, freqs, ids = [], [], []
        for obs in ann:
            if not obs.value["voiced"] or obs.value["frequency"] == 0:
                continue
            times.append(_obs_seconds(obs.time))
            freqs.append(float(obs.value["frequency"]))
            ids.append(int(obs.value.get("index", 0)))

        if times:
            pitch[s], voiced[s], note_ids[s] = _resample_string_to_grid(
                np.asarray(times, dtype=np.float64),
                np.asarray(freqs, dtype=np.float64),
                np.asarray(ids, dtype=np.int64),
                n_frames,
                hop_size_seconds,
            )

    return pitch, voiced, note_ids


def _resample_string_to_grid(
    times: np.ndarray,
    freqs: np.ndarray,
    note_ids: np.ndarray,
    n_frames: int,
    hop_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample one string's observations onto the uniform target grid.

    Observations are segmented into notes (by JAMS note index, and further
    split wherever a time gap exceeds twice the native hop so that a missing
    index field can never merge two notes across a rest). Each note's
    log-frequency curve is linearly interpolated onto the grid points that
    fall within its time span; those frames are marked voiced and tagged with
    a per-string note id (contiguous 0, 1, 2, …); unvoiced frames stay -1.
    """
    pitch    = np.zeros(n_frames, dtype=np.float32)
    voiced   = np.zeros(n_frames, dtype=bool)
    note_grid = np.full(n_frames, -1, dtype=np.int32)

    order    = np.argsort(times, kind="stable")
    times    = times[order]
    freqs    = freqs[order]
    note_ids = note_ids[order]

    grid = np.arange(n_frames) * hop_s
    tol  = hop_s / 2.0
    gap  = 2.0 * _NATIVE_HOP_S

    # Note boundaries: index change OR a rest longer than `gap`.
    if len(times) > 1:
        splits = np.where((np.diff(note_ids) != 0) | (np.diff(times) > gap))[0] + 1
    else:
        splits = np.array([], dtype=np.int64)

    for note_no, seg in enumerate(np.split(np.arange(len(times)), splits)):
        t = times[seg]
        f = freqs[seg]
        t0, t1 = t[0], t[-1]

        sel = np.where((grid >= t0 - tol) & (grid <= t1 + tol))[0]
        if sel.size == 0:
            # A note shorter than one frame lands between grid points: snap it
            # to the nearest frame so single-observation notes aren't dropped.
            k = int(round(t0 / hop_s))
            if 0 <= k < n_frames:
                pitch[k]     = np.float32(f[0])
                voiced[k]    = True
                note_grid[k] = note_no
            continue

        log_hz          = np.interp(grid[sel], t, np.log2(f))
        pitch[sel]      = np.exp2(log_hz).astype(np.float32)
        voiced[sel]     = True
        note_grid[sel]  = note_no

    return pitch, voiced, note_grid


# ---------------------------------------------------------------------------
# Overhang removal
# ---------------------------------------------------------------------------

def remove_pitch_overhangs(
    pitch: np.ndarray,
    voiced: np.ndarray,
    note_ids: np.ndarray,
    divider: int = 5,
    threshold_cents: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Silence frames at the end of notes where pitch drifts beyond threshold.

    Guitar strings decay after plucking, and the pitch often wanders in the
    final moments of a note before falling silent. These 'overhang' frames
    are technically voiced but carry unreliable pitch information; removing
    them gives the model cleaner targets.

    Notes are recovered from ``note_ids`` (the per-frame note tags produced by
    :func:`extract_pitch_note_arrays_jams`), so exact JAMS note boundaries are
    honoured — two adjacent notes on one string are never merged, which would
    otherwise pollute the body-mean reference and risk silencing a legitimate
    note's tail. For each note, the mean pitch over its body (the first
    1 - 1/divider of its frames) is the reference; any tail frame deviating
    more than threshold_cents from it is silenced. Frames within threshold are
    kept even if they lie in the tail.
    """
    pitch  = pitch.copy()
    voiced = voiced.copy()

    for s in range(voiced.shape[0]):
        ids = note_ids[s]
        for note in np.unique(ids[ids >= 0]):
            frames = np.where(ids == note)[0]
            n = frames.size
            if n < divider:
                continue

            tail_start   = frames[0] + (n - n // divider)
            body_frames  = frames[frames < tail_start]
            body         = pitch[s, body_frames]
            body         = body[body > 0]
            if body.size == 0:
                continue
            body_mean = float(body.mean())

            tail_frames = frames[frames >= tail_start]
            tail_vals   = pitch[s, tail_frames]
            with np.errstate(divide="ignore", invalid="ignore"):
                deviation = np.abs(1200.0 * np.log2(tail_vals / body_mean))
            drop_idx = tail_frames[(tail_vals > 0) & (deviation > threshold_cents)]

            pitch[s, drop_idx]  = 0.0
            voiced[s, drop_idx] = False

    return pitch, voiced
