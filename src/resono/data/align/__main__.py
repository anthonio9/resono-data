"""CLI for label alignment against the audio.

    python -m resono.data.align measure --dataset NAME [--cache-dir DIR]
                                        [--sample-rate SR] [--hop-size HOP]
                                        [--workers N]
    python -m resono.data.align apply   --dataset NAME --output-root DIR
                                        --mode {constant,per-track,snap}
                                        [--shift-frames N | --shift-ms MS]
                                        [--cache-dir DIR] [--sample-rate SR]
                                        [--hop-size HOP] [--workers N] [--force]

measure prints each track's label shift and the constant correction it
suggests. apply writes OUTPUT_ROOT/NAME/: the same tracks with corrected
labels (audio hard-linked) and an alignment.json describing the change.
Training reads one cache root for all its datasets, so a dataset that needs
no correction can be put next to the others with --mode constant
--shift-frames 0.
"""
import argparse
import sys
from pathlib import Path

from resono.data.align import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m resono.data.align",
                                     description="Align labels with the note attacks")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("measure", "Measure each track's label shift"),
                       ("apply", "Write a corrected copy of a dataset")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--dataset", required=True)
        p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
        p.add_argument("--sample-rate", type=int, default=22050)
        p.add_argument("--hop-size", type=int, default=256)
        p.add_argument("--workers", type=int, default=None)
        if name == "apply":
            p.add_argument("--output-root", type=Path, required=True)
            p.add_argument("--mode", choices=pipeline.MODES, required=True)
            shift = p.add_mutually_exclusive_group()
            shift.add_argument("--shift-frames", type=int, default=None,
                               help="constant mode: frames to move (negative = earlier)")
            shift.add_argument("--shift-ms", type=float, default=None,
                               help="constant mode: milliseconds, rounded to frames")
            p.add_argument("--force", action="store_true",
                           help="align a dataset that was already aligned")
    args = parser.parse_args()

    dataset_dir = args.cache_dir / args.dataset
    try:
        if args.command == "measure":
            priors, dataset_prior = pipeline.measure(dataset_dir, args.sample_rate,
                                                     args.hop_size, args.workers)
            print(f"{args.dataset}: "
                  + pipeline.summary(priors, dataset_prior, args.sample_rate, args.hop_size))
            return
        given = args.shift_frames is not None or args.shift_ms is not None
        if (args.mode == "constant") != given:
            sys.exit("error: --shift-frames or --shift-ms is required with --mode "
                     "constant, and only there")
        frames = args.shift_frames
        if args.shift_ms is not None:
            frames = round(args.shift_ms / (args.hop_size / args.sample_rate * 1000))
        record = pipeline.apply(dataset_dir, args.output_root / args.dataset,
                                args.sample_rate, args.hop_size, args.mode,
                                shift_frames=frames or 0, workers=args.workers,
                                force=args.force)
        print(f"{args.dataset} -> {args.output_root / args.dataset}: "
              + pipeline.apply_summary(record))
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
