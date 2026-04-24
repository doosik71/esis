"""Dataset adapters for EndoVis data sources."""

from esis.datasets.base import BaseDatasetAdapter
from esis.datasets.endovis17 import EndoVis17Adapter
from esis.datasets.endovis18 import EndoVis18Adapter
from esis.datasets.inspection import ensure_dataset_index, export_dataset_debug, load_dataset_index, summarize_index, validate_index
from esis.datasets.registry import DATASET_REGISTRY, build_all_dataset_indexes, build_dataset_index, create_dataset_adapter
from esis.datasets.schema import DatasetAdapterError, DatasetIndex, DatasetSample

__all__ = [
    "BaseDatasetAdapter",
    "DATASET_REGISTRY",
    "DatasetAdapterError",
    "DatasetIndex",
    "DatasetSample",
    "EndoVis17Adapter",
    "EndoVis18Adapter",
    "build_all_dataset_indexes",
    "build_dataset_index",
    "create_dataset_adapter",
    "ensure_dataset_index",
    "export_dataset_debug",
    "load_dataset_index",
    "summarize_index",
    "validate_index",
]
