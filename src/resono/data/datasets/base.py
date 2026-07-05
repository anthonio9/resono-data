from abc import ABC, abstractmethod
from pathlib import Path


class DatasetModule(ABC):
    @staticmethod
    @abstractmethod
    def download(raw_dir: Path) -> None:
        """Download raw dataset files."""

    @staticmethod
    @abstractmethod
    def preprocess(raw_dir: Path, cache_dir: Path, **kwargs) -> None:
        """Convert raw files to .npy cache."""

    @staticmethod
    @abstractmethod
    def partition(cache_dir: Path, partitions_dir: Path, **kwargs) -> None:
        """Write train/valid/test split JSON."""
