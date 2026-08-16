"""GuitarSet as heard by the guitar's own pickup.

The same 360 performances as :mod:`guitarset`, recorded simultaneously through
the instrument's magnetic pickup rather than an air microphone, under one set
of annotations. Everything about the labels is therefore identical; only the
audio differs, so this module delegates all of it and changes two things: which
file it reads, and which cache it writes to.

It exists as a dataset rather than a flag so that a training mix can name both
sources at once — mic for the corpus every published result uses, pickup for
the signal a plugged-in guitar actually produces — and so that evaluation can
be moved to the deployment domain without rebuilding anything.
"""
from resono.data.datasets.guitarset_pickup.download import download
from resono.data.datasets.guitarset_pickup.partition import partition
from resono.data.datasets.guitarset_pickup.preprocess import preprocess

DATASET_NAME = "guitarset-pickup"

__all__ = ["download", "preprocess", "partition", "DATASET_NAME"]
