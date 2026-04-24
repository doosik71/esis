from __future__ import annotations

from pathlib import Path

from esis.datasets.base import BaseDatasetAdapter
from esis.datasets.endovis15 import EndoVis15Adapter
from esis.datasets.endovis17 import EndoVis17Adapter
from esis.datasets.endovis18 import EndoVis18Adapter
from esis.datasets.endovis19 import EndoVis19Adapter
from esis.datasets.schema import DatasetIndex

DATASET_REGISTRY: dict[str, type[BaseDatasetAdapter]] = {
    "endovis15": EndoVis15Adapter,
    "endovis17": EndoVis17Adapter,
    "endovis18": EndoVis18Adapter,
    "endovis19": EndoVis19Adapter,
}


def create_dataset_adapter(
    name: str,
    dataset_root: str | Path,
    project_root: str | Path | None = None,
) -> BaseDatasetAdapter:
    try:
        adapter_cls = DATASET_REGISTRY[name.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset '{name}'. Supported datasets: {supported}") from exc
    return adapter_cls(dataset_root=dataset_root, project_root=project_root)


def build_dataset_index(
    name: str,
    dataset_root: str | Path,
    project_root: str | Path | None = None,
) -> DatasetIndex:
    return create_dataset_adapter(name, dataset_root, project_root).build_index()


def build_all_dataset_indexes(
    dataset_roots: dict[str, str | Path],
    project_root: str | Path | None = None,
) -> dict[str, DatasetIndex]:
    return {
        name: build_dataset_index(name, dataset_root, project_root)
        for name, dataset_root in dataset_roots.items()
    }
