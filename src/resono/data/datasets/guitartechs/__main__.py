"""CLI for the Guitar-TECHS pipeline.

Usage:
    python -m resono.data.datasets.guitartechs download   [--raw-dir DIR]
                                                          [--archives NAME [NAME ...]]
                                                          [--no-progress-bar]
    python -m resono.data.datasets.guitartechs preprocess [--raw-dir DIR] [--cache-dir DIR]
                                                          [--sample-rate SR] [--hop-size H]
                                                          [--audio-source {directinput,micamp,ego,exo}]
                                                          [--onset-latency-ms MS]
                                                          [--merge-gap-ms MS]
                                                          [--tuning-offset-cents C]
                                                          [--include-all | --exclude NAME [NAME ...]]
                                                          [--no-progress-bar]
    python -m resono.data.datasets.guitartechs partition  [--cache-dir DIR] [--partitions-dir DIR]
                                                          [--held-out-player {P1,P2,P3}]

Electric guitar, three players, three guitars, annotated by a Fishman Triple
Play pickup. Labels are MIDI note events, so pitch is per-note rather than
continuous; see guitartechs.preprocess for what that costs and which takes are
excluded because of it.
"""
import argparse
from pathlib import Path

from resono.data.datasets.guitartechs.download import ARCHIVES, download
from resono.data.datasets.guitartechs.midi import MERGE_GAP_MS, ONSET_LATENCY_MS
from resono.data.datasets.guitartechs.partition import PLAYERS, partition
from resono.data.datasets.guitartechs.preprocess import (
    AUDIO_SOURCES,
    DEFAULT_EXCLUDE,
    preprocess,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m resono.data.datasets.guitartechs",
        description="Guitar-TECHS data pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Download from Zenodo (record 14963133)")
    dl.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    dl.add_argument("--archives", nargs="+", default=None, choices=sorted(ARCHIVES),
                    help="Which archives to fetch; default all nine (~4.1 GB)")
    dl.add_argument("--no-progress-bar", dest="progress",
                    action="store_false", default=True)

    pp = sub.add_parser("preprocess", help="Convert raw files to .npy cache")
    pp.add_argument("--raw-dir",     type=Path, default=Path("data/raw"))
    pp.add_argument("--cache-dir",   type=Path, default=Path("data/cache"))
    pp.add_argument("--sample-rate", type=int,  default=22050)
    pp.add_argument("--hop-size",    type=int,  default=256)
    pp.add_argument("--audio-source", choices=sorted(AUDIO_SOURCES), default="directinput",
                    help="Which synchronised recording to cache (default the "
                         "raw pickup signal)")
    pp.add_argument("--onset-latency-ms", type=float, default=ONSET_LATENCY_MS,
                    help="Undo the pickup's detection delay; measured at ~23 ms")
    pp.add_argument("--merge-gap-ms", type=float, default=MERGE_GAP_MS,
                    help="Notes this close on one string are the same pluck, "
                         "its pitch having crossed a semitone boundary")
    pp.add_argument("--tuning-offset-cents", type=float, default=0.0,
                    help="Added to every labelled pitch; the guitars are not "
                         "tuned to A440 (measured +12c median on P1)")
    group = pp.add_mutually_exclusive_group()
    group.add_argument("--exclude", nargs="+", default=list(DEFAULT_EXCLUDE),
                       help=f"Take names to skip (default {' '.join(DEFAULT_EXCLUDE)})")
    group.add_argument("--include-all", action="store_true", default=False,
                       help="Keep every take, including those whose labels omit "
                            "the technique they are named for")
    pp.add_argument("--no-progress-bar", dest="progress",
                    action="store_false", default=True)

    pt = sub.add_parser("partition", help="Write the train/valid/test split")
    pt.add_argument("--cache-dir",      type=Path, default=Path("data/cache"))
    pt.add_argument("--partitions-dir", type=Path, default=Path("data/partitions"))
    pt.add_argument("--held-out-player", choices=PLAYERS, default=None,
                    help="Hold this player out; default puts everything in "
                         "train, since evaluation happens on another dataset")

    args = parser.parse_args()

    if args.command == "download":
        download(args.raw_dir, archives=args.archives, progress=args.progress)
    elif args.command == "preprocess":
        preprocess(
            args.raw_dir, args.cache_dir,
            sample_rate=args.sample_rate, hop_size=args.hop_size,
            audio_source=args.audio_source,
            onset_latency_ms=args.onset_latency_ms,
            merge_gap_ms=args.merge_gap_ms,
            tuning_offset_cents=args.tuning_offset_cents,
            exclude=() if args.include_all else tuple(args.exclude),
            progress=args.progress,
        )
    elif args.command == "partition":
        partition(args.cache_dir, args.partitions_dir,
                  held_out_player=args.held_out_player)


if __name__ == "__main__":
    main()
