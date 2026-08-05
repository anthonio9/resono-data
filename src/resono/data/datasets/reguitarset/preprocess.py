"""Build the reguitarset cache: GuitarSet labels with FCNF0++ tails.

The cache format is identical to guitarset's — same four arrays, same grid
convention — so everything downstream reads it without knowing the difference.
What differs is where each note's tail pitch comes from.

Audio and annotations are read from the shared raw directory; the FCNF0++
estimates come from the cache written by ``reguitarset track-f0``.
"""
import json
from pathlib import Path

import jams
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from resono.data.datasets.guitarset.preprocess import extract_pitch_note_arrays_jams
from resono.data.datasets.reguitarset.f0 import (
    NATIVE_HOP_SIZE,
    NATIVE_SAMPLE_RATE,
    load_f0,
)
from resono.data.datasets.reguitarset.relabel import (
    format_summary,
    relabel_tails,
    summarise,
    write_audit,
)

DATASET_NAME = "reguitarset"

AUDIT_FILENAME   = "relabel-audit.csv"
SUMMARY_FILENAME = "relabel-summary.json"


def preprocess(
    raw_dir: Path,
    cache_dir: Path,
    f0_dir: Path = Path("data/f0-fcnf0"),
    sample_rate: int = NATIVE_SAMPLE_RATE,
    hop_size: int = NATIVE_HOP_SIZE,
    tail_policy: str = "track",
    offset_policy: str = "both",
    divider: int = 5,
    periodicity_threshold: float = 0.3,
    max_extend_frames: int = 64,
    median_filter: int = 0,
    progress: bool = True,
) -> None:
    """Convert raw GuitarSet files to a .npy cache with relabelled tails.

    Produces per-track files in cache_dir/reguitarset/:
        {stem}-audio.npy   float32  (N_samples,)
        {stem}-pitch.npy   float32  (6, N_frames)   Hz, 0 = unvoiced
        {stem}-voiced.npy  bool     (6, N_frames)
        {stem}-onset.npy   bool     (6, N_frames)   True on note-start frames

    plus two files describing the relabelling itself:
        relabel-audit.csv    one row per note — see relabel.NoteAudit
        relabel-summary.json aggregates over every note

    Parameters
    ----------
    sample_rate, hop_size:
        Target grid. Defaults are GuitarSet's native 5.805 ms (11025 Hz, hop
        64) rather than guitarset's 22050/256, which halves the resolution the
        annotations actually carry.
    f0_dir:
        The F0 cache from 'reguitarset track-f0'. Its grid must match, or be an
        exact integer divisor of, the target grid.

    See :func:`relabel.relabel_tails` for the relabelling parameters.
    """
    gset_root = Path(raw_dir) / "guitarset"
    out_dir   = Path(cache_dir) / DATASET_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    hop_s = hop_size / sample_rate

    audio_index = {
        p.stem.replace("_mic", ""): p for p in gset_root.rglob("*_mic.wav")
    }
    jams_index = {p.stem: p for p in gset_root.rglob("*.jams")}

    # The F0 cache drives the loop, not the audio directory. It is the scarcest
    # of the three inputs — 'track-f0 --limit' deliberately produces a partial
    # one for timing pilots — so iterating it means there is never a track to
    # skip, and a pilot costs a pilot's worth of work rather than a full pass.
    stems = sorted(p.name[: -len("-f0.npy")] for p in Path(f0_dir).glob("*-f0.npy"))
    if not stems:
        raise FileNotFoundError(
            f"No *-f0.npy in {f0_dir}. Run 'reguitarset track-f0' first."
        )

    audits = []
    processed = 0

    for stem in tqdm(stems, desc="Preprocessing", unit="track", disable=not progress):
        jams_file = jams_index.get(stem)
        audio_file = audio_index.get(stem)
        if jams_file is None or audio_file is None:
            missing = "JAMS" if jams_file is None else "mic audio"
            tqdm.write(f"  warning: no {missing} for {stem}, skipping")
            continue

        # --- audio ---
        # The mic mix, exactly as guitarset caches it. The hexaphonic channels
        # inform the labels but must not become the model's input: resono's
        # problem is separating strings from a single mixed signal, and
        # training on pre-separated audio would define that problem away.
        audio, sr = sf.read(audio_file)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        audio = audio.astype(np.float32)

        # --- labels ---
        jam = jams.load(str(jams_file))
        n_frames = len(audio) // hop_size
        pitch, voiced, note_ids = extract_pitch_note_arrays_jams(jam, hop_s, n_frames)

        f0, periodicity = load_f0(f0_dir, stem, sample_rate, hop_size, n_frames)

        # Onsets come from the pristine note_ids, before any tail work: note
        # starts are not what is being corrected, and deriving them here keeps
        # them identical to guitarset's even where a tail moves.
        onset = np.zeros_like(voiced, dtype=bool)
        onset[:, 0] = note_ids[:, 0] != -1
        onset[:, 1:] = (note_ids[:, 1:] != note_ids[:, :-1]) & (note_ids[:, 1:] != -1)

        pitch, voiced, track_audits = relabel_tails(
            pitch, voiced, note_ids, f0, periodicity,
            stem=stem,
            hop_seconds=hop_s,
            tail_policy=tail_policy,
            offset_policy=offset_policy,
            divider=divider,
            periodicity_threshold=periodicity_threshold,
            max_extend_frames=max_extend_frames,
            median_filter=median_filter,
        )
        audits.extend(track_audits)

        np.save(out_dir / f"{stem}-audio.npy",  audio)
        np.save(out_dir / f"{stem}-pitch.npy",  pitch)
        np.save(out_dir / f"{stem}-voiced.npy", voiced)
        np.save(out_dir / f"{stem}-onset.npy",  onset)
        processed += 1

    if processed == 0:
        raise FileNotFoundError(
            f"None of the {len(stems)} tracks in {f0_dir} had both mic audio and "
            f"a JAMS annotation under {gset_root}."
        )

    summary = summarise(audits)
    summary["settings"] = {
        "sample_rate": sample_rate,
        "hop_size": hop_size,
        "hop_seconds": hop_s,
        "tail_policy": tail_policy,
        "offset_policy": offset_policy,
        "divider": divider,
        "periodicity_threshold": periodicity_threshold,
        "max_extend_frames": max_extend_frames,
        "median_filter": median_filter,
    }

    write_audit(audits, out_dir / AUDIT_FILENAME)
    with open(out_dir / SUMMARY_FILENAME, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPreprocessed {processed} tracks → {out_dir}")
    print(f"\nRelabelling ({tail_policy} tails, {offset_policy} offsets):")
    print(format_summary(summary))
    print(
        f"\n  audit   → {out_dir / AUDIT_FILENAME}"
        f"\n  summary → {out_dir / SUMMARY_FILENAME}"
        "\n\nSort the audit by |body_cents_diff| and inspect the worst rows with:"
        "\n  python -m resono.data plot --dataset reguitarset "
        "--compare-dataset guitarset --stem STEM --start T_START"
    )
