from __future__ import annotations

import argparse
import json
from pathlib import Path

from esis import __version__
from esis.datasets import export_dataset_debug
from esis.gui.dataset_preview import launch_dataset_preview_app
from esis.segmentation import available_segmenters
from esis.segmentation.runner import SegmentationRunSelection, run_segmentation_selection
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

    segment_parser = subparsers.add_parser("segment", help="Segmentation execution utilities.")
    segment_subparsers = segment_parser.add_subparsers(dest="segment_command")

    run_parser = segment_subparsers.add_parser("run", help="Run one backend on one image, one split, or one sequence.")
    run_parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(default_dataset_roots().keys()),
        help="Dataset name to use.",
    )
    run_parser.add_argument(
        "--backend",
        required=True,
        choices=available_segmenters(),
        help="Segmentation backend to run.",
    )
    run_parser.add_argument("--sample-id", help="Run on one sample id.")
    run_parser.add_argument("--split", help="Run on all samples in one split.")
    run_parser.add_argument("--sequence-id", help="Run on all samples in one sequence.")
    run_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of selected samples.",
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

    if args.command == "segment" and args.segment_command == "run":
        summary = run_segmentation_selection(
            SegmentationRunSelection(
                dataset_name=args.dataset,
                backend_name=args.backend,
                sample_id=args.sample_id,
                split=args.split,
                sequence_id=args.sequence_id,
                limit=args.limit,
            )
        )
        print(json.dumps(summary, indent=2))
        return 0

    print("ESIS project is initialized.")
    print("Use this entry point to add dataset, segmentation, tracking, and evaluation commands.")
    print("Run `python main.py --show-paths` to inspect the project layout.")
    return 0
