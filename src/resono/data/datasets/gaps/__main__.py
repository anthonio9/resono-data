"""CLI for the GAPS dataset pipeline.

Usage:
    python -m resono.data.datasets.gaps download   [--raw-dir DIR] [--no-audio]
                                                   [--no-progress-bar]
    python -m resono.data.datasets.gaps preprocess [--raw-dir DIR] [--cache-dir DIR]
                                                   [--sample-rate SR] [--hop-size H]
                                                   [--tolerance SECONDS]
                                                   [--no-progress-bar]
    python -m resono.data.datasets.gaps partition  [--cache-dir DIR] [--partitions-dir DIR]
                                                   [--raw-dir DIR]
                                                   [--valid-fraction F] [--seed N]

Progress bars (download files, preprocess tracks) are on by default; disable
with --no-progress-bar.
"""
import argparse
from pathlib import Path

from resono.data.datasets.gaps.download import download
from resono.data.datasets.gaps.preprocess import preprocess
from resono.data.datasets.gaps.partition import partition


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m resono.data.datasets.gaps",
        description="GAPS data pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # download
    dl = sub.add_parser("download", help="Download from HuggingFace")
    dl.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    dl.add_argument("--no-audio", dest="audio",
                    action="store_false", default=True)
    dl.add_argument("--no-progress-bar", dest="progress",
                    action="store_false", default=True)

    # preprocess
    pp = sub.add_parser("preprocess", help="Convert raw files to .npy cache")
    pp.add_argument("--raw-dir",     type=Path,  default=Path("data/raw"))
    pp.add_argument("--cache-dir",   type=Path,  default=Path("data/cache"))
    pp.add_argument("--sample-rate", type=int,   default=22050)
    pp.add_argument("--hop-size",    type=int,   default=256)
    pp.add_argument("--tolerance",   type=float, default=2.0)
    pp.add_argument("--no-progress-bar", dest="progress",
                    action="store_false", default=True)

    # partition
    pt = sub.add_parser("partition", help="Create train/valid/test split")
    pt.add_argument("--cache-dir",      type=Path,  default=Path("data/cache"))
    pt.add_argument("--partitions-dir", type=Path,  default=Path("data/partitions"))
    pt.add_argument("--raw-dir",        type=Path,  default=Path("data/raw"))
    pt.add_argument("--valid-fraction", type=float, default=0.1)
    pt.add_argument("--seed",           type=int,   default=42)

    args = parser.parse_args()

    if args.command == "download":
        download(args.raw_dir, audio=args.audio, progress=args.progress)
    elif args.command == "preprocess":
        preprocess(
            args.raw_dir, args.cache_dir, args.sample_rate, args.hop_size,
            tolerance=args.tolerance,
            progress=args.progress,
        )
    elif args.command == "partition":
        partition(
            args.cache_dir, args.partitions_dir,
            raw_dir=args.raw_dir,
            valid_fraction=args.valid_fraction,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
