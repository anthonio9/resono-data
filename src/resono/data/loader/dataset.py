import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    """Memory-mapped dataset over one or more preprocessed .npy caches.

    Each preprocessed track contributes these files in cache_dir/{dataset}/:
        {stem}-audio.npy   float32  (N_samples,)
        {stem}-pitch.npy   float32  (6, N_frames) or (N_frames,)
        {stem}-voiced.npy  bool     same shape as pitch
        {stem}-onset.npy   bool     same shape as pitch (True on note starts)

    Items are fixed-size chunks of consecutive frames. The sampler controls
    which chunks are drawn and in what order — __getitem__ just loads whatever
    global frame index it receives.

    Parameters
    ----------
    datasets : list[str]
        Dataset names to load (e.g. ['guitarset']). Each must have a matching
        {name}.json partition file and a cache_dir/{name}/ directory.
    partitions_dir : Path
        Directory containing the {name}.json partition files.
    cache_dir : Path
        Root cache directory. Each dataset occupies cache_dir/{name}/.
    split : str
        Which partition to load: 'train', 'valid', or 'test'.
    hop_size : int
        Audio samples per pitch frame, fixed at preprocessing time.
        Determines the audio-to-pitch alignment: frame f starts at sample
        f * hop_size. Example: hop_size=256 at 22050 Hz ≈ 11.6 ms per frame.
    window_frames : int
        Number of consecutive pitch frames per returned item — the temporal
        context the model receives in one forward pass. Audio samples per item
        = window_frames * hop_size. Example: window_frames=32, hop_size=256
        → 8192 samples ≈ 370 ms of audio.
    build_onset_idx : bool
        If True, also precompute onset_idx: the subset of voiced_idx where at
        least one string transitions from unvoiced to voiced at that frame.
        Disabled by default; voiced_idx is sufficient for most training runs.
    """

    def __init__(
        self,
        datasets: list[str],
        partitions_dir: Path,
        cache_dir: Path,
        split: str,
        hop_size: int = 256,
        window_frames: int = 32,
        build_onset_idx: bool = False,
    ) -> None:
        self.hop_size      = hop_size
        self.window_frames = window_frames

        # (prefix_path, n_frames, stem) per track.
        self._items: list[tuple[Path, int, str]] = []

        for name in datasets:
            partition_file = Path(partitions_dir) / f"{name}.json"
            with open(partition_file) as f:
                partition = json.load(f)

            dataset_dir = Path(cache_dir) / name
            for stem in partition[split]:
                pitch    = np.load(dataset_dir / f"{stem}-pitch.npy", mmap_mode="r")
                n_frames = pitch.shape[-1]
                if n_frames >= window_frames:
                    self._items.append((dataset_dir / stem, n_frames, stem))

        # Cumulative frame counts enable O(log n) global-index → item lookup.
        self._cum: list[int] = [0]
        for _, n, _ in self._items:
            self._cum.append(self._cum[-1] + n)

        # Lazily-populated per-process cache of open memmaps, keyed by prefix.
        # Left empty here on purpose: with DataLoader workers the dataset is
        # forked, and memmaps are opened on first access *inside* each worker
        # rather than shared across the fork boundary.
        self._cache: dict[str, dict[str, np.ndarray]] = {}

        # Precompute index pools by scanning voiced.npy files.
        self.voiced_idx: list[int] = []
        self.onset_idx:  list[int] = []
        self._build_indices(build_onset_idx)

    # ------------------------------------------------------------------
    # Index pool construction
    # ------------------------------------------------------------------

    def _build_indices(self, build_onset: bool) -> None:
        for item_idx, (prefix, n_frames, _) in enumerate(self._items):
            voiced = np.load(f"{prefix}-voiced.npy", mmap_mode="r")

            # Collapse strings: a frame is voiced if any string is active.
            any_voiced = voiced.any(axis=0) if voiced.ndim == 2 else voiced.astype(bool)

            offset = self._cum[item_idx]

            # All voiced frames are valid starts. Windows that extend past the
            # track end are zero-padded in __getitem__, so boundary frames are
            # included — the model trains on cold endings and cold starts alike.
            valid_voiced = np.where(any_voiced)[0]
            self.voiced_idx.extend((valid_voiced + offset).tolist())

            if build_onset:
                onset_mask     = np.zeros(n_frames, dtype=bool)
                onset_mask[0]  = any_voiced[0]
                onset_mask[1:] = any_voiced[1:] & ~any_voiced[:-1]
                valid_onsets   = np.where(onset_mask)[0]
                self.onset_idx.extend((valid_onsets + offset).tolist())

    # ------------------------------------------------------------------
    # Inference index pool
    # ------------------------------------------------------------------

    def inference_idx(self) -> list[int]:
        """Non-overlapping chunk starts covering every track sequentially.

        Chunks tile each track end to end with stride window_frames. The final
        chunk of a track may run past its end; __getitem__ zero-pads that tail,
        so every real frame — including silence and the last partial window —
        is covered exactly once. This lets a full-track prediction be
        reconstructed without gaps. Pair with EpochSampler(shuffle=False), and
        discard predictions beyond each track's true length when reassembling.
        """
        indices = []
        for item_idx, (_, n_frames, _) in enumerate(self._items):
            offset = self._cum[item_idx]
            for f in range(0, n_frames, self.window_frames):
                indices.append(offset + f)
        return indices

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._cum[-1]

    def _get_arrays(self, prefix: Path) -> dict[str, np.ndarray]:
        """Return cached memmaps for a track, opening them on first access."""
        key = str(prefix)
        arrays = self._cache.get(key)
        if arrays is None:
            arrays = {
                "audio":  np.load(f"{prefix}-audio.npy",  mmap_mode="r"),
                "pitch":  np.load(f"{prefix}-pitch.npy",  mmap_mode="r"),
                "voiced": np.load(f"{prefix}-voiced.npy", mmap_mode="r"),
                "onset":  np.load(f"{prefix}-onset.npy",  mmap_mode="r"),
            }
            self._cache[key] = arrays
        return arrays

    def __getitem__(self, idx: int) -> dict:
        item_idx               = bisect.bisect_right(self._cum, idx) - 1
        frame                  = idx - self._cum[item_idx]
        prefix, n_frames, stem = self._items[item_idx]

        W = self.window_frames
        H = self.hop_size

        arrays = self._get_arrays(prefix)
        pitch, voiced, onset, audio = (
            arrays["pitch"], arrays["voiced"], arrays["onset"], arrays["audio"])

        end_frame = frame + W

        if end_frame <= n_frames:
            pitch_slice  = np.array(pitch[...,  frame:end_frame])
            voiced_slice = np.array(voiced[..., frame:end_frame])
            onset_slice  = np.array(onset[...,  frame:end_frame])
            audio_slice  = np.array(audio[frame * H : end_frame * H])
        else:
            # Window extends past the track end: zero-pad the tail so that
            # boundary frames (including track onsets when context < window)
            # are seen during training. Audio is clipped to the frame grid
            # (n_frames * H) first — the raw file may carry up to H-1 extra
            # samples that must not leak into the fixed-size window.
            pad_frames    = end_frame - n_frames
            pitch_avail   = np.array(pitch[...,  frame:])
            voiced_avail  = np.array(voiced[..., frame:])
            onset_avail   = np.array(onset[...,  frame:])
            audio_avail   = np.array(audio[frame * H : n_frames * H])
            pitch_slice   = np.concatenate([pitch_avail,  np.zeros((*pitch_avail.shape[:-1],  pad_frames), dtype=np.float32)], axis=-1)
            voiced_slice  = np.concatenate([voiced_avail, np.zeros((*voiced_avail.shape[:-1], pad_frames), dtype=bool)],       axis=-1)
            onset_slice   = np.concatenate([onset_avail,  np.zeros((*onset_avail.shape[:-1],  pad_frames), dtype=bool)],       axis=-1)
            audio_slice   = np.concatenate([audio_avail,  np.zeros(pad_frames * H,                         dtype=np.float32)])

        return {
            "audio":  torch.from_numpy(audio_slice).float(),
            "pitch":  torch.from_numpy(pitch_slice).float(),
            "voiced": torch.from_numpy(voiced_slice),
            "onset":  torch.from_numpy(onset_slice),
            "stem":   stem,
        }
