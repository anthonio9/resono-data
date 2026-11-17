"""Tests for label alignment: the label edits, the snapper, and apply.

The snapper cases use Karplus-Strong plucks with known attack samples, the
same kind of scene its thresholds were tuned on.
"""
import json

import numpy as np
import pytest

from resono.data.align import pipeline
from resono.data.align.labels import move_note_start, notes_of, translate
from resono.data.align.snapper import (
    MIN_ANCHORS, Note, Prior, estimate_prior, snap_notes,
)

SR, HOP = 44100, 256
FRAME_MS = HOP / SR * 1000


def _pluck(f0, seconds, rng):
    """Karplus-Strong pluck, peak 1; -> (signal, its true f0)."""
    period = max(2, int(round(SR / f0 - 0.5)))
    y = np.zeros(int(seconds * SR) + period + 1)
    y[:period + 1] = rng.uniform(-1, 1, period + 1)
    loss = 10 ** (-3 / (2.5 * SR / (period + 0.5)))
    j = period + 1
    while j < len(y):
        k = min(period, len(y) - j)
        y[j:j + k] = loss * 0.5 * (y[j - period:j - period + k] + y[j - period - 1:j - period - 1 + k])
        j += k
    y = y[:int(seconds * SR)]
    return y / np.abs(y).max(), SR / (period + 0.5)


def _track(rng, n_notes=MIN_ANCHORS + 6, late_ms=17.0):
    """Isolated plucks 0.4 s apart, labelled late_ms after their attacks.
    -> audio, labels (onset, voiced, pitch), true attack samples, strings."""
    seconds = 0.5 + 0.4 * n_notes
    audio = 1e-4 * rng.standard_normal(int(seconds * SR))
    frames = int(seconds * SR) // HOP
    onset = np.zeros((6, frames), bool)
    voiced = np.zeros((6, frames), bool)
    pitch = np.zeros((6, frames), np.float32)
    attacks, strings = [], []
    for i in range(n_notes):
        string = i % 6
        attack = int((0.3 + 0.4 * i) * SR) + int(rng.integers(0, HOP))
        sig, f0 = _pluck(110 * 2 ** (rng.integers(0, 24) / 12), 0.3, rng)
        audio[attack:attack + len(sig)] += 0.3 * sig
        label = int((attack + late_ms / 1000 * SR) // HOP)
        end = label + int(0.25 * SR / HOP)
        onset[string, label] = True
        voiced[string, label:end] = True
        pitch[string, label:end] = f0
        attacks.append(attack)
        strings.append(string)
    return audio.astype(np.float32), (onset, voiced, pitch), attacks, strings


# ---------------------------------------------------------------------------
# Label edits
# ---------------------------------------------------------------------------

def _one_note(n=20, start=5, end=12, hz=110.0):
    onset = np.zeros((6, n), bool)
    voiced = np.zeros((6, n), bool)
    pitch = np.zeros((6, n), np.float32)
    onset[2, start] = True
    voiced[2, start:end] = True
    pitch[2, start:end] = hz
    return onset, voiced, pitch


def test_translate_moves_onset_voicing_and_pitch_together():
    onset, voiced, pitch = translate(*_one_note(), -3)
    assert np.flatnonzero(onset[2]).tolist() == [2]
    assert np.flatnonzero(voiced[2]).tolist() == list(range(2, 9))
    assert (pitch[2, 2:9] == 110).all() and (pitch[2, 9:] == 0).all()
    assert onset.dtype == bool and pitch.dtype == np.float32


def test_translate_later_leaves_the_start_unvoiced():
    onset, voiced, pitch = translate(*_one_note(start=0, end=6), 2)
    assert np.flatnonzero(onset[2]).tolist() == [2]
    assert not voiced[2, :2].any() and (pitch[2, :2] == 0).all()


def test_translate_restarts_a_note_whose_onset_falls_off_the_start():
    onset, voiced, _ = translate(*_one_note(start=1, end=10), -3)
    assert np.flatnonzero(onset[2]).tolist() == [0]
    assert voiced[2, 0]


def test_translate_keeps_one_dimensional_labels():
    onset, voiced, pitch = (a[2] for a in _one_note())
    out = translate(onset, voiced, pitch, -1)
    assert [a.shape for a in out] == [(20,)] * 3
    assert np.flatnonzero(out[0]).tolist() == [4]


def test_move_start_earlier_voices_the_gained_frames_at_the_note_pitch():
    onset, voiced, pitch = _one_note()
    assert move_note_start(onset, voiced, pitch, 2, 5, 3)
    assert np.flatnonzero(onset[2]).tolist() == [3]
    assert voiced[2, 3:12].all() and (pitch[2, 3:5] == 110).all()


def test_move_start_earlier_cuts_the_note_ringing_before_it():
    onset, voiced, pitch = _one_note(start=5, end=12, hz=110.0)
    onset[2, 0] = True
    voiced[2, 0:5] = True
    pitch[2, 0:5] = 220.0
    assert move_note_start(onset, voiced, pitch, 2, 5, 3)
    assert (pitch[2, 0:3] == 220).all() and (pitch[2, 3:12] == 110).all()
    assert np.flatnonzero(onset[2]).tolist() == [0, 3]


def test_move_start_never_crosses_another_onset():
    onset, voiced, pitch = _one_note(start=5)
    onset[2, 3] = True
    assert not move_note_start(onset, voiced, pitch, 2, 5, 2)
    assert onset[2, 5]


def test_move_start_later_unvoices_the_lost_frames():
    onset, voiced, pitch = _one_note()
    assert move_note_start(onset, voiced, pitch, 2, 5, 7)
    assert np.flatnonzero(onset[2]).tolist() == [7]
    assert not voiced[2, 5:7].any() and (pitch[2, 5:7] == 0).all()


def test_move_start_later_refuses_to_erase_the_note():
    onset, voiced, pitch = _one_note(start=5, end=8)
    assert not move_note_start(onset, voiced, pitch, 2, 5, 8)
    assert onset[2, 5]


def test_notes_of_reads_the_first_labelled_pitch():
    onset, voiced, pitch = _one_note()
    pitch[2, 5] = 0                  # pitch tracks often start a frame late
    assert notes_of(onset, pitch) == [Note(2, 5, 110.0)]


# ---------------------------------------------------------------------------
# Snapper
# ---------------------------------------------------------------------------

def test_prior_finds_the_label_lateness():
    rng = np.random.default_rng(0)
    audio, (onset, _, pitch), _, _ = _track(rng, late_ms=17.0)
    prior = estimate_prior(audio, SR, HOP, notes_of(onset, pitch))
    assert prior is not None and prior.anchors >= MIN_ANCHORS
    # label centres sit 17 ms + half a frame (on average) after the attack
    assert -17 - FRAME_MS <= prior.shift_ms <= -17 + 1


def test_prior_needs_enough_lone_notes():
    rng = np.random.default_rng(1)
    audio, (onset, _, pitch), _, _ = _track(rng, n_notes=MIN_ANCHORS - 5)
    assert estimate_prior(audio, SR, HOP, notes_of(onset, pitch)) is None


def test_snap_lands_on_the_attack():
    rng = np.random.default_rng(2)
    audio, (onset, _, pitch), attacks, _ = _track(rng, late_ms=25.0)
    notes = notes_of(onset, pitch)
    snaps = snap_notes(audio, SR, HOP, notes, Prior(-25.0, 8.0, 0))
    assert [s.status for s in snaps].count("precise") >= len(snaps) - 2
    errors = [abs(s.sample - a) / SR * 1000 for s, a in zip(snaps, sorted(attacks))
              if s.sample is not None]
    assert np.median(errors) < 1.0


def test_strum_notes_keep_their_label_when_the_labels_are_tight():
    rng = np.random.default_rng(3)
    audio = 1e-4 * rng.standard_normal(SR)
    onset = np.zeros((6, SR // HOP), bool)
    pitch = np.zeros((6, SR // HOP), np.float32)
    for string, hz in enumerate((82.4, 123.5, 164.8, 207.7, 246.9, 329.6)):
        attack = int(0.3 * SR) + string * int(0.01 * SR)
        sig, f0 = _pluck(hz, 0.5, rng)
        audio[attack:attack + len(sig)] += 0.2 * sig
        frame = (attack + int(0.017 * SR)) // HOP
        onset[string, frame] = True
        pitch[string, frame:frame + 40] = f0
    snaps = snap_notes(audio, SR, HOP, notes_of(onset, pitch), Prior(-17.0, 6.0, 0))
    assert {s.status for s in snaps} == {"kept"}


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path):
    rng = np.random.default_rng(4)
    audio, (onset, voiced, pitch), attacks, strings = _track(rng, late_ms=17.0)
    dataset = tmp_path / "cache" / "toy"
    dataset.mkdir(parents=True)
    for kind, array in (("audio", audio), ("onset", onset), ("voiced", voiced),
                        ("pitch", pitch)):
        np.save(dataset / f"take-{kind}.npy", array)
    (dataset / "manifest.json").write_text("{}")
    return dataset, attacks, strings


def test_apply_constant_moves_every_label(cache, tmp_path):
    dataset, _, _ = cache
    out = tmp_path / "aligned" / "toy"
    pipeline.apply(dataset, out, SR, HOP, "constant", shift_frames=-3, workers=1)
    before = np.flatnonzero(np.load(dataset / "take-onset.npy").any(0))
    after = np.flatnonzero(np.load(out / "take-onset.npy").any(0))
    assert (after == before - 3).all()
    assert (out / "take-audio.npy").exists() and (out / "manifest.json").exists()
    assert json.loads((out / "alignment.json").read_text())["tracks"]["take"]["shift_frames"] == -3


def test_apply_snap_puts_onsets_on_the_attack_frames(cache, tmp_path):
    dataset, attacks, strings = cache
    out = tmp_path / "aligned" / "toy"
    record = pipeline.apply(dataset, out, SR, HOP, "snap", workers=2)
    onset = np.load(out / "take-onset.npy")
    hits = sum(onset[s, a // HOP] for s, a in zip(strings, attacks))
    assert hits >= len(attacks) - 2
    assert record["tracks"]["take"]["prior"]["anchors"] >= MIN_ANCHORS
    # voicing still starts with each onset
    voiced = np.load(out / "take-voiced.npy")
    assert voiced[onset].all()


def test_apply_refuses_to_align_twice(cache, tmp_path):
    dataset, _, _ = cache
    once = tmp_path / "once" / "toy"
    pipeline.apply(dataset, once, SR, HOP, "constant", shift_frames=-3, workers=1)
    with pytest.raises(ValueError, match="already aligned"):
        pipeline.apply(once, tmp_path / "twice" / "toy", SR, HOP, "constant",
                       shift_frames=-3, workers=1)


def test_apply_refuses_to_overwrite_its_source(cache):
    dataset, _, _ = cache
    with pytest.raises(ValueError, match="different directory"):
        pipeline.apply(dataset, dataset, SR, HOP, "constant", shift_frames=0, workers=1)
