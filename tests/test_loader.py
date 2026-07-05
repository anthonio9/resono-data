import torch

from resono.data.loader.loader import get_loader


def make_loader(fd, split="train", seed=0, **kwargs):
    return get_loader(
        fd["datasets"], fd["partitions_dir"], fd["cache_dir"],
        split=split,
        hop_size=fd["hop_size"],
        window_frames=fd["window_frames"],
        batch_size=4,
        num_workers=0,
        seed=seed,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Batch shapes
# ---------------------------------------------------------------------------

def test_train_batch_shapes(fake_dataset):
    fd     = fake_dataset
    loader = make_loader(fd)
    batch  = next(iter(loader))

    W, H = fd["window_frames"], fd["hop_size"]
    assert batch["audio"].shape  == (4, W * H)
    assert batch["pitch"].shape  == (4, 6, W)
    assert batch["voiced"].shape == (4, 6, W)
    assert len(batch["stem"])    == 4


def test_inference_batch_shapes(fake_dataset):
    fd     = fake_dataset
    loader = make_loader(fd, split="valid")
    batch  = next(iter(loader))

    W, H = fd["window_frames"], fd["hop_size"]
    assert batch["audio"].shape[-1] == W * H
    assert batch["pitch"].shape[-1] == W


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_same_order(fake_dataset):
    """Two loaders with the same seed must yield identical first batches."""
    fd = fake_dataset
    b1 = next(iter(make_loader(fd, seed=42)))
    b2 = next(iter(make_loader(fd, seed=42)))
    assert torch.equal(b1["audio"], b2["audio"])


def test_different_seeds_different_order(fake_dataset):
    fd = fake_dataset
    b1 = next(iter(make_loader(fd, seed=0)))
    b2 = next(iter(make_loader(fd, seed=99)))
    assert not torch.equal(b1["audio"], b2["audio"])


# ---------------------------------------------------------------------------
# Voiced guarantee
# ---------------------------------------------------------------------------

def test_train_batches_are_voiced(fake_dataset):
    """Every training batch item must contain at least one voiced frame."""
    loader = make_loader(fake_dataset)
    for batch in loader:
        # voiced: (B, 6, W) — at least one True per item
        per_item = batch["voiced"].any(dim=-1).any(dim=-1)   # (B,)
        assert per_item.all(), "Training batch contains a fully silent item"


# ---------------------------------------------------------------------------
# Onset mode
# ---------------------------------------------------------------------------

def test_onset_loader_runs(fake_dataset):
    loader = make_loader(fake_dataset, use_onset_idx=True)
    batch  = next(iter(loader))
    assert "audio" in batch
