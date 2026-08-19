"""CLI for the GOAT dataset pipeline.

GOAT is distributed by request (Zenodo record 15690894) so there is no
download command; extract the archive under --raw-dir and run:

    python -m resono.data.datasets.goat preprocess [--raw-dir DIR] [--cache-dir DIR]
                                                   [--audio-source {di,amp1..amp5,gp}]
                                                   [--alignment-threshold F]
                                                   [--disable-unaligned]
                                                   [--tolerance SECONDS]
    python -m resono.data.datasets.goat partition  [--cache-dir DIR]
                                                   [--partitions-dir DIR]
                                                   [--held-out-player P]
"""
import argparse
from pathlib import Path

from resono.data.datasets.goat.preprocess import AUDIO_SOURCES, preprocess
from resono.data.datasets.goat.partition import partition


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m resono.data.datasets.goat", description="GOAT data pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("preprocess", help="Convert raw files to .npy cache")
    pp.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    pp.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    pp.add_argument("--sample-rate", type=int, default=22050)
    pp.add_argument("--hop-size", type=int, default=256)
    pp.add_argument("--audio-source", choices=sorted(AUDIO_SOURCES), default="di",
                    help="Which recording to cache (default the raw DI pickup)")
    pp.add_argument("--alignment-threshold", type=float, default=0.5,
                    help="Minimum alignment_f_measure_fine; 0.5 drops only "
                         "outright failures, higher values discard dense takes")
    pp.add_argument("--disable-unaligned", action="store_true", default=False,
                    help="Skip the 20 takes with no fine-aligned MIDI "
                         "(kept by default)")
    pp.add_argument("--tolerance", type=float, default=2.0)
    pp.add_argument("--no-progress-bar", dest="progress",
                    action="store_false", default=True)

    pa = sub.add_parser("partition", help="Write the train/valid/test split")
    pa.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    pa.add_argument("--partitions-dir", type=Path, default=Path("data/partitions"))
    pa.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    pa.add_argument("--held-out-player", type=str, default=None)

    args = parser.parse_args()
    if args.command == "preprocess":
        preprocess(
            args.raw_dir, args.cache_dir,
            sample_rate=args.sample_rate, hop_size=args.hop_size,
            audio_source=args.audio_source,
            alignment_threshold=args.alignment_threshold,
            disable_unaligned=args.disable_unaligned,
            tolerance=args.tolerance, progress=args.progress,
        )
    else:
        partition(args.cache_dir, args.partitions_dir,
                  held_out_player=args.held_out_player, raw_dir=args.raw_dir)


if __name__ == "__main__":
    main()
