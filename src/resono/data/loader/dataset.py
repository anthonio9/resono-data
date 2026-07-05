import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    """Memory-mapped dataset over one or more preprocessed .npy caches.

    Each preprocessed track contributes three files in cache_dir/{dataset}/:
        {stem}-audio.npy   float32  (N_samples,)
        {stem}-pitch.npy   float32  (6, N_frames) or (N_frames,)
        {stem}-voiced.npy  bool     same shape as pitch

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

        Includes silent chunks so the full-track prediction can be
        reconstructed without gaps. Pair with EpochSampler(shuffle=False).
        """
        indices = []
        for item_idx, (_, n_frames, _) in enumerate(self._items):
            offset = self._cum[item_idx]
            for f in range(0, n_frames - self.window_frames + 1, self.window_frames):
                indices.append(offset + f)
        return indices

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._cum[-1]

    def __getitem__(self, idx: int) -> dict:
        item_idx              = bisect.bisect_right(self._cum, idx) - 1
        frame                 = idx - self._cum[item_idx]
        prefix, n_frames, stem = self._items[item_idx]

        W = self.window_frames
        H = self.hop_size

        pitch  = np.load(f"{prefix}-pitch.npy",  mmap_mode="r")
        voiced = np.load(f"{prefix}-voiced.npy", mmap_mode="r")
        audio  = np.load(f"{prefix}-audio.npy",  mmap_mode="r")

        end_frame = frame + W

        if end_frame <= n_frames:
            pitch_s  = np.array(pitch[...,  frame:end_frame])
            voiced_s = np.array(voiced[..., frame:end_frame])
            audio_s  = np.array(audio[frame * H : end_frame * H])
        else:
            # Window extends past the track end: zero-pad the tail so that
            # boundary frames (including track onsets when context < window)
            # are seen during training.
            pad_f    = end_frame - n_frames
            p        = np.array(pitch[...,  frame:])
            v        = np.array(voiced[..., frame:])
            a        = np.array(audio[frame * H:])
            pitch_s  = np.concatenate([p, np.zeros((*p.shape[:-1], pad_f), dtype=np.float32)], axis=-1)
            voiced_s = np.concatenate([v, np.zeros((*v.shape[:-1], pad_f), dtype=bool)],       axis=-1)
            audio_s  = np.concatenate([a, np.zeros(pad_f * H,              dtype=np.float32)])

        return {
            "audio":  torch.from_numpy(audio_s).float(),
            "pitch":  torch.from_numpy(pitch_s).float(),
            "voiced": torch.from_numpy(voiced_s),
            "stem":   stem,
        }
