"""GOAT: paired direct-input electric guitar audio and Guitar Pro tablature."""
from resono.data.datasets.goat.preprocess import DATASET_NAME, preprocess
from resono.data.datasets.goat.partition import partition

__all__ = ["DATASET_NAME", "preprocess", "partition"]
