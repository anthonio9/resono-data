from pathlib import Path

from torch.utils.data import DataLoader

from resono.data.loader.dataset import Dataset
from resono.data.loader.sampler import EpochSampler


def get_loader(
    datasets: list[str],
    partitions_dir: Path,
    cache_dir: Path,
    split: str,
    hop_size: int = 256,
    window_frames: int = 32,
    use_onset_idx: bool = False,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 0,
) -> DataLoader:
    """Create a DataLoader for the given split.

    Training returns shuffled chunks drawn only from voiced (or onset) frames.
    Validation and test return sequential non-overlapping chunks covering every
    track in full, including silent regions, so predictions can be reconstructed
    track by track.

    Parameters
    ----------
    datasets : list[str]
        Dataset names to load (e.g. ['guitarset']).
    partitions_dir : Path
        Directory containing {name}.json partition files.
    cache_dir : Path
        Root cache directory; each dataset occupies cache_dir/{name}/.
    split : str
        'train', 'valid', or 'test'.
    hop_size : int
        Audio samples per pitch frame, fixed at preprocessing time.
        Must match the value used during preprocessing.
        Example: hop_size=256 at 22050 Hz ≈ 11.6 ms per frame.
    window_frames : int
        Consecutive frames per batch item — the model's temporal context window.
        Audio samples per item = window_frames * hop_size.
        Example: window_frames=32, hop_size=256 → 8192 samples ≈ 370 ms.
    use_onset_idx : bool
        Training only. If True, restrict sampling to onset frames (at least one
        string transitioning unvoiced→voiced). Default False: sample from all
        voiced frames, which covers onsets, sustained notes, and everything in
        between.
    batch_size : int
        Items per batch. One batch therefore contains batch_size * window_frames
        total frames of labelled audio.
    num_workers : int
        Worker processes for parallel data loading.
    seed : int
        Base seed for the EpochSampler. The permutation for epoch e is derived
        from seed + e, keeping runs reproducible across restarts.
    """
    dataset = Dataset(
        datasets, partitions_dir, cache_dir, split,
        hop_size=hop_size,
        window_frames=window_frames,
        build_onset_idx=use_onset_idx,
    )

    if split == "train":
        indices = dataset.onset_idx if use_onset_idx else dataset.voiced_idx
        sampler = EpochSampler(indices, seed=seed, shuffle=True)
    else:
        indices = dataset.inference_idx()
        sampler = EpochSampler(indices, seed=seed, shuffle=False)

    import torch
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
