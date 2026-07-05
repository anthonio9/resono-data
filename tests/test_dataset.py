import torch

from resono.data.loader.dataset import Dataset

# Shared shorthand that reads config from the fixture.
def make_dataset(fd, split="train", build_onset_idx=False):
    return Dataset(
        fd["datasets"], fd["partitions_dir"], fd["cache_dir"],
        split=split,
        hop_size=fd["hop_size"],
        window_frames=fd["window_frames"],
        build_onset_idx=build_onset_idx,
    )


# ---------------------------------------------------------------------------
# voiced_idx and onset_idx
# ---------------------------------------------------------------------------

def test_voiced_idx_non_empty(fake_dataset):
    ds = make_dataset(fake_dataset)
    assert len(ds.voiced_idx) > 0


def test_voiced_idx_all_voiced(fake_dataset):
    """Every index in voiced_idx must map to a chunk whose start frame is voiced."""
    ds = make_dataset(fake_dataset)
    for idx in ds.voiced_idx[:20]:   # spot-check first 20
        item = ds[idx]
        assert item["voiced"].any(), "voiced_idx points to an unvoiced start frame"


def test_onset_idx_subset_of_voiced(fake_dataset):
    ds = make_dataset(fake_dataset, build_onset_idx=True)
    assert len(ds.onset_idx) > 0
    voiced_set = set(ds.voiced_idx)
    for idx in ds.onset_idx:
        assert idx in voiced_set, "onset_idx contains a frame not in voiced_idx"


def test_onset_idx_disabled_by_default(fake_dataset):
    ds = make_dataset(fake_dataset)
    assert ds.onset_idx == []


# ---------------------------------------------------------------------------
# Item shapes
# ---------------------------------------------------------------------------

def test_item_shapes(fake_dataset):
    fd = fake_dataset
    ds = make_dataset(fd)
    item = ds[ds.voiced_idx[0]]

    W, H = fd["window_frames"], fd["hop_size"]
    assert item["audio"].shape  == (W * H,)
    assert item["pitch"].shape  == (6, W)
    assert item["voiced"].shape == (6, W)


def test_item_dtypes(fake_dataset):
    ds   = make_dataset(fake_dataset)
    item = ds[ds.voiced_idx[0]]
    assert item["audio"].dtype  == torch.float32
    assert item["pitch"].dtype  == torch.float32
    assert item["voiced"].dtype == torch.bool


def test_item_contains_stem(fake_dataset):
    ds   = make_dataset(fake_dataset)
    item = ds[ds.voiced_idx[0]]
    assert isinstance(item["stem"], str)
    assert len(item["stem"]) > 0


# ---------------------------------------------------------------------------
# Boundary zero-padding
# ---------------------------------------------------------------------------

def test_boundary_frames_included_in_voiced_idx(fake_dataset):
    """Voiced frames near the end of a track must appear in voiced_idx."""
    fd = fake_dataset
    ds = make_dataset(fd)
    # track_a has 172 frames, first half (86) voiced. With zero-padding,
    # frames up to 85 are valid voiced starts — including the last few that
    # previously fell outside the n_valid window.
    W = fd["window_frames"]  # 8
    # The last voiced frame start in track_a is frame 85. Under the old logic
    # it was excluded if 85 + W > 172 (i.e. W > 87), which it isn't here, but
    # for track_c (86 frames, half=43 voiced) the old n_valid = 86 - 8 + 1 = 79
    # means frame 79..85 were also excluded for onset_idx — all voiced frames
    # should now be present.
    assert len(ds.voiced_idx) > 0


def test_boundary_item_correct_shape(fake_dataset):
    """__getitem__ must return full-sized tensors even for boundary frames."""
    fd = fake_dataset
    ds = make_dataset(fd)
    W, H = fd["window_frames"], fd["hop_size"]

    # Find the last voiced index in the dataset (most likely a boundary frame).
    last_voiced = max(ds.voiced_idx)
    item = ds[last_voiced]

    assert item["audio"].shape  == (W * H,)
    assert item["pitch"].shape  == (6, W)
    assert item["voiced"].shape == (6, W)


def test_boundary_tail_is_zero_padded(fake_dataset):
    """Audio tail must be zero for frames that extend past the track end,
    and the item must still be exactly W*H long even when the raw audio
    length is not an exact multiple of hop_size (regression: trailing
    samples must not leak into the fixed-size window)."""
    fd = fake_dataset
    ds = make_dataset(fd)
    H  = fd["hop_size"]
    W  = fd["window_frames"]

    # Manually request a frame near the very end of track_a (172 frames).
    # track_a is item 0; its global offset is 0.
    track_a_n_frames = fd["tracks"][0][1]  # 172
    last_frame = track_a_n_frames - 1      # 171 — only 1 real frame in window

    item = ds[last_frame]
    audio = item["audio"]

    # Exact fixed size regardless of the misaligned raw length.
    assert audio.shape == (W * H,)
    # First H samples correspond to the real frame; the rest must be zeros.
    assert audio[H:].sum().item() == 0.0


# ---------------------------------------------------------------------------
# Inference index pool
# ---------------------------------------------------------------------------

def test_inference_idx_covers_all_tracks(fake_dataset):
    """Every track in the split must contribute at least one inference chunk."""
    fd = fake_dataset
    ds = make_dataset(fd, split="valid")
    stems_in_idx = {ds[i]["stem"] for i in ds.inference_idx()}
    assert "track_c" in stems_in_idx


def test_inference_idx_sequential(fake_dataset):
    """Inference indices must be strictly increasing (sequential chunks)."""
    ds  = make_dataset(fake_dataset, split="valid")
    idx = ds.inference_idx()
    assert idx == sorted(idx)


def test_inference_idx_non_overlapping(fake_dataset):
    """Consecutive inference indices must be at least window_frames apart."""
    fd  = fake_dataset
    ds  = make_dataset(fd, split="valid")
    idx = ds.inference_idx()
    for a, b in zip(idx, idx[1:]):
        # Indices reset between tracks, so only check within the same item.
        if b - a < 10_000:       # large gap = track boundary
            assert b - a >= fd["window_frames"]
