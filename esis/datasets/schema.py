from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DatasetSample:
    dataset_name: str
    split: str
    sample_id: str
    sequence_id: str
    modality: str
    image_path: str | None = None
    label_path: str | None = None
    video_path: str | None = None
    frame_index: int | None = None
    timestamp_ms: int | None = None
    width: int | None = None
    height: int | None = None
    annotations: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatasetIndex:
    dataset_name: str
    dataset_root: str
    index_path: str
    sample_schema_version: str
    sample_count: int
    sequence_count: int
    sequences: list[dict[str, Any]]
    samples: list[DatasetSample]
    metadata_files: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["samples"] = [sample.to_dict() for sample in self.samples]
        return payload


class DatasetAdapterError(RuntimeError):
    """Raised when a dataset adapter cannot parse the target dataset."""


def ensure_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
