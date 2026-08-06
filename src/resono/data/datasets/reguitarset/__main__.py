"""CLI for the reguitarset pipeline — GuitarSet with FCNF0++ note tails.

Usage:
    python -m resono.data.datasets.reguitarset download   [--raw-dir DIR] [--no-progress-bar]
    python -m resono.data.datasets.reguitarset track-f0   [--raw-dir DIR] [--f0-dir DIR]
                                                          [--sample-rate SR] [--hop-size H]
                                                          [--batch-size N] [--gpu N] [--limit N]
                                                          [--workers N] [--overwrite]
    python -m resono.data.datasets.reguitarset preprocess [--raw-dir DIR] [--cache-dir DIR]
                                                          [--f0-dir DIR]
                                                          [--sample-rate SR] [--hop-size H]
                                                          [--tail-policy {track,hold}]
                                                          [--offset-policy {both,trim,extend,none}]
                                                          [--divider N]
                                                          [--periodicity-threshold P]
                                                          [--max-extend-frames N]
                                                          [--median-filter N]
    python -m resono.data.datasets.reguitarset partition  [--cache-dir DIR] [--partitions-dir DIR]
                                                          [--no-player-split] [--seed N]
                                                          [--val-players P [P ...]]
                                                          [--test-players P [P ...]]
    python -m resono.data.datasets.reguitarset cv-folds   [--cache-dir DIR] [--partitions-dir DIR]

Run them in that order: track-f0 is the expensive step and preprocess reads its
cache, so the relabelling parameters can be re-tuned without re-running it.
"""
import argparse
from pathlib import Path

from resono.data.datasets.reguitarset.download import download
from resono.data.datasets.reguitarset.f0 import (
    NATIVE_HOP_SIZE,
    NATIVE_SAMPLE_RATE,
    track_f0,
)
from resono.data.datasets.reguitarset.partition import cv_folds, partition
from resono.data.datasets.reguitarset.preprocess import preprocess
from resono.data.datasets.reguitarset.relabel import OFFSET_POLICIES, TAIL_POLICIES

_DEFAULT_F0_DIR = Path("data/f0-fcnf0")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m resono.data.datasets.reguitarset",
        description="GuitarSet relabelled with FCNF0++ note tails",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # download
    dl = sub.add_parser("download", help="Download GuitarSet plus hexaphonic audio")
    dl.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    dl.add_argument("--no-progress-bar", dest="progress",
                    action="store_false", default=True)

    # track-f0
    tf = sub.add_parser("track-f0", help="Run FCNF0++ over the hexaphonic channels")
    tf.add_argument("--raw-dir",     type=Path, default=Path("data/raw"))
    tf.add_argument("--f0-dir",      type=Path, default=_DEFAULT_F0_DIR)
    tf.add_argument("--sample-rate", type=int,  default=NATIVE_SAMPLE_RATE)
    tf.add_argument("--hop-size",    type=int,  default=NATIVE_HOP_SIZE)
    tf.add_argument("--batch-size",  type=int,  default=128,
                    help="Frames per forward pass. Runtime is flat in this; "
                         "memory is linear (0.67 GB at 128, 4.21 GB at 2048)")
    tf.add_argument("--gpu",         type=int,  default=None)
    tf.add_argument("--limit",       type=int,  default=None,
                    help="Process only the first N tracks (for a timing pilot)")
    tf.add_argument("--workers",     type=int,  default=3,
                    help="Tracks in parallel, ~0.7 GB each. Gain saturates "
                         "near 3 (36.9s/channel at 1, 21.8s at 3, 21.2s at 6)")
    tf.add_argument("--overwrite",   action="store_true", default=False,
                    help="Recompute tracks already present; otherwise the "
                         "run resumes where it left off")
    tf.add_argument("--no-progress-bar", dest="progress",
                    action="store_false", default=True)

    # preprocess
    pp = sub.add_parser("preprocess", help="Build the cache with relabelled tails")
    pp.add_argument("--raw-dir",     type=Path, default=Path("data/raw"))
    pp.add_argument("--cache-dir",   type=Path, default=Path("data/cache"))
    pp.add_argument("--f0-dir",      type=Path, default=_DEFAULT_F0_DIR)
    pp.add_argument("--sample-rate", type=int,  default=NATIVE_SAMPLE_RATE)
    pp.add_argument("--hop-size",    type=int,  default=NATIVE_HOP_SIZE)
    pp.add_argument("--tail-policy",   choices=TAIL_POLICIES,   default="track")
    pp.add_argument("--offset-policy", choices=OFFSET_POLICIES, default="both")
    pp.add_argument("--divider",               type=int,   default=5)
    pp.add_argument("--periodicity-threshold", type=float, default=0.3)
    pp.add_argument("--max-extend-frames",     type=int,   default=64)
    pp.add_argument("--median-filter",         type=int,   default=0,
                    help="Odd smoothing window for 'track' tails; 0 (default) "
                         "disables it — penn's Viterbi decoding already "
                         "removes what it would remove")
    pp.add_argument("--no-progress-bar", dest="progress",
                    action="store_false", default=True)

    # partition
    pt = sub.add_parser("partition", help="Create train/valid/test split")
    pt.add_argument("--cache-dir",      type=Path, default=Path("data/cache"))
    pt.add_argument("--partitions-dir", type=Path, default=Path("data/partitions"))
    pt.add_argument("--no-player-split", dest="split_by_player",
                    action="store_false", default=True)
    pt.add_argument("--val-players",  nargs="+", default=None)
    pt.add_argument("--test-players", nargs="+", default=None)
    pt.add_argument("--seed", type=int, default=42)

    # cv-folds
    cv = sub.add_parser("cv-folds", help="Write all 6 cross-validation fold JSONs")
    cv.add_argument("--cache-dir",      type=Path, default=Path("data/cache"))
    cv.add_argument("--partitions-dir", type=Path, default=Path("data/partitions"))

    args = parser.parse_args()

    if args.command == "download":
        download(args.raw_dir, progress=args.progress)

    elif args.command == "track-f0":
        track_f0(
            args.raw_dir, args.f0_dir,
            sample_rate=args.sample_rate, hop_size=args.hop_size,
            batch_size=args.batch_size, gpu=args.gpu, limit=args.limit,
            workers=args.workers, overwrite=args.overwrite,
            progress=args.progress,
        )

    elif args.command == "preprocess":
        preprocess(
            args.raw_dir, args.cache_dir, args.f0_dir,
            sample_rate=args.sample_rate, hop_size=args.hop_size,
            tail_policy=args.tail_policy,
            offset_policy=args.offset_policy,
            divider=args.divider,
            periodicity_threshold=args.periodicity_threshold,
            max_extend_frames=args.max_extend_frames,
            median_filter=args.median_filter,
            progress=args.progress,
        )

    elif args.command == "partition":
        partition(
            args.cache_dir, args.partitions_dir,
            split_by_player=args.split_by_player,
            val_players=args.val_players,
            test_players=args.test_players,
            seed=args.seed,
        )

    elif args.command == "cv-folds":
        cv_folds(args.cache_dir, args.partitions_dir)


if __name__ == "__main__":
    main()
