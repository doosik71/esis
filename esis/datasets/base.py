from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

from esis.datasets.schema import DatasetIndex, DatasetSample


class BaseDatasetAdapter(ABC):
    dataset_name: str = ""
    index_file_name: str = ""

    def __init__(self, dataset_root: str | Path, project_root: str | Path | None = None) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.project_root = (
            Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[2]
        )
        self.index_root = self.project_root / "temp" / "index"

    @abstractmethod
    def collect_samples(self) -> list[DatasetSample]:
        """Collect dataset samples."""

    def build_index(self) -> DatasetIndex:
        samples = self.collect_samples()
        sequence_ids = sorted({sample.sequence_id for sample in samples})
        split_counts = Counter(sample.split for sample in samples)
        modality_counts = Counter(sample.modality for sample in samples)
        sequence_map: dict[str, dict[str, object]] = {}
        for sample in samples:
            sequence = sequence_map.setdefault(
                sample.sequence_id,
                {
                    "sequence_id": sample.sequence_id,
                    "split": sample.split,
                    "modality": sample.modality,
                    "sample_count": 0,
                },
            )
            sequence["sample_count"] = int(sequence["sample_count"]) + 1

        index_path = self.index_root / self.index_file_name
        index = DatasetIndex(
            dataset_name=self.dataset_name,
            dataset_root=str(self.dataset_root),
            index_path=str(index_path),
            sample_schema_version="1.0",
            sample_count=len(samples),
            sequence_count=len(sequence_ids),
            sequences=list(sequence_map.values()),
            samples=samples,
            metadata_files=self.find_metadata_files(),
            summary={
                "split_counts": dict(split_counts),
                "modality_counts": dict(modality_counts),
            },
        )
        self.save_index(index)
        return index

    def save_index(self, index: DatasetIndex) -> Path:
        self.index_root.mkdir(parents=True, exist_ok=True)
        index_path = Path(index.index_path)
        index_path.write_text(json.dumps(index.to_dict(), indent=2), encoding="utf-8")
        return index_path

    def find_metadata_files(self) -> list[str]:
        files = []
        for suffix in ("*.json", "*.csv", "*.tsv", "*.txt", "*.md", "*.pdf"):
            for path in self.dataset_root.rglob(suffix):
                if path.is_file():
                    files.append(str(path))
        return sorted(files)

    def parse_tsv_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            return [dict(row) for row in reader]
