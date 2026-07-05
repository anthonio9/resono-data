"""Top-level CLI dispatcher.

Each dataset exposes its own pipeline under resono.data.datasets.<name>:
    python -m resono.data.datasets.guitarset download
    python -m resono.data.datasets.guitarset preprocess
    python -m resono.data.datasets.guitarset partition
    python -m resono.data.datasets.guitarset cv-folds
"""
print(__doc__)
