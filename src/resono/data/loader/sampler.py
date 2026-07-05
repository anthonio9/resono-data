import torch
from torch.utils.data import Sampler


class EpochSampler(Sampler):
    """Reproducible per-epoch shuffle over a fixed index pool.

    An epoch is one full pass through the training data. This sampler changes
    its shuffle permutation each epoch so the model sees items in a different
    order every pass — without this, the same sequence of batches repeats every
    epoch, which can cause the model to overfit to the ordering rather than the
    content. The permutation is still deterministic: the same seed + epoch pair
    always produces the same order, which makes runs reproducible.

    Draws from a pre-filtered list of indices (e.g. voiced or onset frames)
    rather than from the full dataset range, so the DataLoader never sees
    silent or invalid chunks unless they are explicitly included in the pool.

    Call set_epoch(epoch) at the start of each epoch to advance the permutation.

    Parameters
    ----------
    indices : list[int]
        Pre-filtered global frame indices to sample from.
    seed : int
        Base seed. The actual seed for epoch e is seed + e, giving a unique
        but reproducible permutation every epoch.
    shuffle : bool
        If False, yields indices in their original order — use for validation
        and test where sequential processing is required.
    """

    def __init__(
        self,
        indices: list[int],
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        self.indices = indices
        self.seed    = seed
        self.shuffle = shuffle
        self._epoch  = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        if not self.shuffle:
            return iter(self.indices)
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        order = torch.randperm(len(self.indices), generator=g).tolist()
        return (self.indices[i] for i in order)
