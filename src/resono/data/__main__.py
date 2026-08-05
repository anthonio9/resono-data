"""Top-level CLI dispatcher.

Each dataset exposes its own pipeline under resono.data.datasets.<name>:
    python -m resono.data.datasets.guitarset download
    python -m resono.data.datasets.guitarset preprocess
    python -m resono.data.datasets.guitarset partition
    python -m resono.data.datasets.guitarset cv-folds

    python -m resono.data.datasets.gaps download
    python -m resono.data.datasets.gaps preprocess
    python -m resono.data.datasets.gaps partition

    python -m resono.data.datasets.reguitarset download
    python -m resono.data.datasets.reguitarset verify-hex
    python -m resono.data.datasets.reguitarset track-f0
    python -m resono.data.datasets.reguitarset preprocess
    python -m resono.data.datasets.reguitarset partition

Inspect preprocessed labels against the audio they describe (any dataset):
    python -m resono.data plot --list-stems
    python -m resono.data plot --dataset gaps --stem 001_mvswc --start 40

Compare two datasets' labels for the same track, to see what a relabelling did:
    python -m resono.data plot --dataset reguitarset --compare-dataset guitarset \\
        --stem 05_Rock1-90-C_solo --start 12.4
"""
import argparse
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "plot":
        print(__doc__)
        return

    parser = argparse.ArgumentParser(
        prog="python -m resono.data plot",
        description="Plot preprocessed labels over the audio they describe",
    )
    parser.add_argument("plot", help=argparse.SUPPRESS)
    parser.add_argument("--dataset",     required=True)
    parser.add_argument("--stem",        default=None,
                        help="Track stem; defaults to the first cached track")
    parser.add_argument("--list-stems",  action="store_true",
                        help="List the dataset's cached tracks and exit")
    parser.add_argument("--cache-dir",   type=Path,  default=Path("data/cache"))
    parser.add_argument("--start",       type=float, default=0.0)
    parser.add_argument("--duration",    type=float, default=10.0)
    parser.add_argument("--sample-rate", type=int,   default=22050)
    parser.add_argument("--hop-size",    type=int,   default=256)
    parser.add_argument("--theme",       choices=["light", "dark"], default="light")
    parser.add_argument("--output",      type=Path,  default=None,
                        help="Write a PNG here instead of opening a window")
    parser.add_argument("--compare-dataset", default=None,
                        help="Overlay this dataset's labels for the same stem")
    parser.add_argument("--compare-cache-dir", type=Path, default=None,
                        help="Cache root for --compare-dataset (default: --cache-dir)")
    args = parser.parse_args()

    # A missing cache is ordinary user error, not a crash: report it as one
    # line rather than a traceback that buries the message under a stack.
    try:
        if args.list_stems:
            list_stems(args.cache_dir, args.dataset, args.sample_rate, args.hop_size)
            return

        from resono.data.plot import plot_track

        stem = args.stem or _stems(args.cache_dir, args.dataset)[0]
        plot_track(
            args.cache_dir, args.dataset, stem,
            start=args.start, duration=args.duration,
            sample_rate=args.sample_rate, hop_size=args.hop_size,
            theme=args.theme, output=args.output,
            compare_dataset=args.compare_dataset,
            compare_cache_dir=args.compare_cache_dir,
        )
    except (FileNotFoundError, ValueError) as error:
        sys.exit(f"error: {error}")

    if args.output is None:
        import matplotlib.pyplot as plt
        plt.show()


def list_stems(
    cache_dir: Path, dataset: str, sample_rate: int = 22050, hop_size: int = 256
) -> None:
    """Print the dataset's cached tracks, with what each one carries."""
    import numpy as np

    root = Path(cache_dir) / dataset
    stems = _stems(cache_dir, dataset)

    print(f"{dataset}: {len(stems)} cached tracks in {root}\n")
    print(f"{'stem':<34} {'duration':>9} {'frames':>8}  onsets")
    total = 0.0
    for stem in stems:
        # mmap reads the .npy header only — no track is loaded to be listed.
        samples = np.load(root / f"{stem}-audio.npy", mmap_mode="r").shape[0]
        frames = np.load(root / f"{stem}-pitch.npy", mmap_mode="r").shape[-1]
        seconds = samples / sample_rate
        total += seconds
        has_onsets = (root / f"{stem}-onset.npy").exists()
        print(
            f"{stem:<34} {seconds / 60:>6.1f}min {frames:>8}"
            f"  {'yes' if has_onsets else '—'}"
        )
    print(f"\n{total / 3600:.2f} h total")


def _stems(cache_dir: Path, dataset: str) -> list[str]:
    """Cached stems for one dataset, newest-format first-come-first-served.

    When the dataset has no cache the error names the ones that do, so a typo
    or a not-yet-preprocessed dataset is self-diagnosing.
    """
    root = Path(cache_dir) / dataset
    stems = sorted(p.stem[:-6] for p in root.glob("*-audio.npy"))
    if stems:
        return stems

    available = sorted(
        directory.name for directory in Path(cache_dir).glob("*")
        if directory.is_dir() and any(directory.glob("*-audio.npy"))
    )
    raise FileNotFoundError(
        f"No preprocessed tracks in {root}. "
        + (f"Cached datasets: {', '.join(available)}." if available
           else f"Nothing is cached under {cache_dir} yet.")
    )


if __name__ == "__main__":
    main()
