from __future__ import annotations

import argparse
from pathlib import Path

from esis import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="esis",
        description="ESIS project entry point for dataset, tracking, and experiment tooling.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Print key project directories and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.show_paths:
        root = Path(__file__).resolve().parents[2]
        for name in ("esis", "data", "temp"):
            print(f"{name}: {root / name}")
        return 0

    print("ESIS project is initialized.")
    print("Use this entry point to add dataset, segmentation, tracking, and evaluation commands.")
    print("Run `python main.py --show-paths` to inspect the project layout.")
    return 0
