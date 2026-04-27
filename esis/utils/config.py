from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_dataset_roots(root: Path | None = None) -> dict[str, Path]:
    base = (root or project_root()) / "data"
    return {
        "endovis17": base / "EndoVis17 Instrument Subchallenge Dataset",
        "endovis18": base / "EndoVis18 Instrument Subchallenge Dataset",
    }


def index_root(root: Path | None = None) -> Path:
    return (root or project_root()) / "temp" / "index"


def debug_root(root: Path | None = None) -> Path:
    return (root or project_root()) / "temp" / "debug"


def cache_root(root: Path | None = None) -> Path:
    return (root or project_root()) / "temp" / "cache"


def checkpoint_root(root: Path | None = None) -> Path:
    return cache_root(root) / "checkpoints"


def runs_root(root: Path | None = None) -> Path:
    return (root or project_root()) / "temp" / "runs"
