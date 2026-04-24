from __future__ import annotations

import argparse
import json
from pathlib import Path

from esis import __version__
from esis.datasets import export_dataset_debug
from esis.gui.dataset_preview import launch_dataset_preview_app
from esis.utils.config import default_dataset_roots


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
    subparsers = parser.add_subparsers(dest="command")

    dataset_parser = subparsers.add_parser("dataset", help="Dataset inspection utilities.")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command")

    inspect_parser = dataset_subparsers.add_parser("inspect", help="Build reports, validations, and previews.")
    inspect_parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(default_dataset_roots().keys()),
        help="Dataset name to inspect.",
    )
    inspect_parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the dataset index before inspection.",
    )
    inspect_parser.add_argument(
        "--preview-count",
        type=int,
        default=6,
        help="Number of previews to export to temp/debug.",
    )
    inspect_parser.add_argument(
        "--validate-count",
        type=int,
        default=64,
        help="Number of samples to validate from the index.",
    )
    dataset_subparsers.add_parser("gui", help="Launch the dataset preview GUI.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.show_paths:
        root = Path(__file__).resolve().parents[2]
        for name in ("esis", "data", "temp"):
            print(f"{name}: {root / name}")
        return 0

    if args.command == "dataset" and args.dataset_command == "inspect":
        outputs = export_dataset_debug(
            dataset_name=args.dataset,
            rebuild_index=args.rebuild_index,
            preview_count=args.preview_count,
            validate_count=args.validate_count,
        )
        print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
        return 0

    if args.command == "dataset" and args.dataset_command == "gui":
        return launch_dataset_preview_app()

    print("ESIS project is initialized.")
    print("Use this entry point to add dataset, segmentation, tracking, and evaluation commands.")
    print("Run `python main.py --show-paths` to inspect the project layout.")
    return 0
