"""Dataset adapters for EndoVis data sources."""

from esis.datasets.base import BaseDatasetAdapter
from esis.datasets.endovis15 import EndoVis15Adapter
from esis.datasets.endovis17 import EndoVis17Adapter
from esis.datasets.endovis18 import EndoVis18Adapter
from esis.datasets.endovis19 import EndoVis19Adapter
from esis.datasets.registry import DATASET_REGISTRY, build_all_dataset_indexes, build_dataset_index, create_dataset_adapter
from esis.datasets.schema import DatasetAdapterError, DatasetIndex, DatasetSample

__all__ = [
    "BaseDatasetAdapter",
    "DATASET_REGISTRY",
    "DatasetAdapterError",
    "DatasetIndex",
    "DatasetSample",
    "EndoVis15Adapter",
    "EndoVis17Adapter",
    "EndoVis18Adapter",
    "EndoVis19Adapter",
    "build_all_dataset_indexes",
    "build_dataset_index",
    "create_dataset_adapter",
]
