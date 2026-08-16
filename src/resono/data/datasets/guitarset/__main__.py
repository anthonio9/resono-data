"""CLI for the GuitarSet dataset pipeline.

Usage:
    python -m resono.data.datasets.guitarset download   [--raw-dir DIR] [--no-progress-bar]
    python -m resono.data.datasets.guitarset preprocess [--raw-dir DIR] [--cache-dir DIR]
                                                        [--sample-rate SR] [--hop-size H]
                                                        [--flatten-tails]
                                                        [--no-progress-bar]
    python -m resono.data.datasets.guitarset partition  [--cache-dir DIR] [--partitions-dir DIR]
                                                        [--no-player-split] [--seed N]
                                                        [--val-players P [P ...]]
                                                        [--test-players P [P ...]]
    python -m resono.data.datasets.guitarset cv-folds   [--cache-dir DIR] [--partitions-dir DIR]

Progress bars (download bytes, preprocess tracks) are on by default; disable
with --no-progress-bar.
"""
import argparse
from pathlib import Path

from resono.data.datasets.guitarset.download import ARCHIVES, download
from resono.data.datasets.guitarset.preprocess import AUDIO_SOURCES, preprocess
from resono.data.datasets.guitarset.partition import cv_folds, partition


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m resono.data.datasets.guitarset",
        description="GuitarSet data pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # download
    dl = sub.add_parser("download", help="Download from Zenodo")
    dl.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    dl.add_argument("--archives", nargs="+", default=None, choices=sorted(ARCHIVES),
                    help="Which archives to fetch; default all three (~1.4 GB)")
    dl.add_argument("--keep-sharp-names", dest="rename_sharp",
                    action="store_false", default=True,
                    help="Keep the archive's own '#' spelling instead of "
                         "rewriting it to 'sharp'. Mixing spellings across "
                         "archives drops tracks silently, so this is on by default")
    dl.add_argument("--no-progress-bar", dest="progress",
                    action="store_false", default=True)

    # preprocess
    pp = sub.add_parser("preprocess", help="Convert raw files to .npy cache")
    pp.add_argument("--raw-dir",                  type=Path,  default=Path("data/raw"))
    pp.add_argument("--cache-dir",                type=Path,  default=Path("data/cache"))
    pp.add_argument("--sample-rate",              type=int,   default=22050)
    pp.add_argument("--hop-size",                 type=int,   default=256)
    pp.add_argument("--remove-overhangs",         action="store_true", default=False)
    pp.add_argument("--overhang-divider",         type=int,   default=5)
    pp.add_argument("--overhang-threshold-cents", type=float, default=15.0)
    pp.add_argument("--flatten-tails",             action="store_true", default=False,
                    help="Hold each note's pre-drop pitch through the pitch "
                         "fall at its end, so the label does not read as a "
                         "MIDI pitch bend. Slides and released bends survive")
    pp.add_argument("--drop-threshold-cents",      type=float, default=25.0,
                    help="How far below the body a frame must sit to count as "
                         "part of the drop (with --flatten-tails)")
    pp.add_argument("--audio-source", choices=sorted(AUDIO_SOURCES), default="mic",
                    help="Which simultaneous recording to cache: the air "
                         "microphone, or the guitar's own pickup. Labels are "
                         "identical either way")
    pp.add_argument("--dataset-name", default=None,
                    help="Cache subdirectory and partition name; defaults to "
                         "guitarset for mic and guitarset-pickup for pickup")
    pp.add_argument("--no-progress-bar", dest="progress",
                    action="store_false", default=True)

    # partition
    pt = sub.add_parser("partition", help="Create train/valid/test split")
    pt.add_argument("--cache-dir",      type=Path,  default=Path("data/cache"))
    pt.add_argument("--partitions-dir", type=Path,  default=Path("data/partitions"))
    pt.add_argument("--name", default="guitarset-mic",
                    help="Dataset name to split, for GuitarSet-derived caches "
                         "sharing these stems and this player numbering")
    pt.add_argument("--no-mirror", dest="mirror_from", action="store_const", const=None,
                    default="guitarset-mic",
                    help="Compute a fresh split instead of copying guitarset-mic's. "
                         "Variants hold the same performances, so splitting "
                         "them independently can leak across train and test")
    pt.add_argument("--no-player-split", dest="split_by_player",
                    action="store_false", default=True)
    pt.add_argument("--val-players",  nargs="+", default=None)
    pt.add_argument("--test-players", nargs="+", default=None)
    pt.add_argument("--seed", type=int, default=42)

    # cv-folds
    cv = sub.add_parser("cv-folds", help="Write all 6 cross-validation fold JSONs")
    cv.add_argument("--cache-dir",      type=Path, default=Path("data/cache"))
    cv.add_argument("--partitions-dir", type=Path, default=Path("data/partitions"))
    cv.add_argument("--name", default="guitarset-mic",
                    help="Dataset to fold, matching preprocess's output name")

    args = parser.parse_args()

    if args.command == "download":
        download(args.raw_dir, archives=args.archives,
                 rename_sharp=args.rename_sharp, progress=args.progress)
    elif args.command == "preprocess":
        preprocess(
            args.raw_dir, args.cache_dir, args.sample_rate, args.hop_size,
            remove_overhangs=args.remove_overhangs,
            overhang_divider=args.overhang_divider,
            overhang_threshold_cents=args.overhang_threshold_cents,
            flatten_tails=args.flatten_tails,
            drop_threshold_cents=args.drop_threshold_cents,
            audio_source=args.audio_source,
            dataset_name=args.dataset_name,
            progress=args.progress,
        )
    elif args.command == "partition":
        partition(
            args.cache_dir, args.partitions_dir,
            split_by_player=args.split_by_player,
            val_players=args.val_players,
            test_players=args.test_players,
            seed=args.seed,
            name=args.name,
            mirror_from=args.mirror_from,
        )
    elif args.command == "cv-folds":
        cv_folds(args.cache_dir, args.partitions_dir, name=args.name)


if __name__ == "__main__":
    main()
