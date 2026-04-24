from __future__ import annotations

from pathlib import Path

from esis.datasets.base import BaseDatasetAdapter
from esis.datasets.schema import DatasetAdapterError, DatasetSample


class EndoVis15Adapter(BaseDatasetAdapter):
    dataset_name = "endovis15"
    index_file_name = "endovis15_index.json"

    def collect_samples(self) -> list[DatasetSample]:
        data_root = self.dataset_root / "data"
        if not data_root.exists():
            raise DatasetAdapterError(f"Missing dataset data directory: {data_root}")

        samples: list[DatasetSample] = []
        for group_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
            nested_root = self._resolve_nested_root(group_dir)
            if nested_root is None:
                continue

            split = self._infer_split(group_dir.name)
            task = "segmentation" if "Segmentation" in group_dir.name else "tracking"
            domain = "robotic" if "Robotic" in group_dir.name else "rigid"

            for sequence_dir in sorted(path for path in nested_root.iterdir() if path.is_dir()):
                sample_id = f"{group_dir.name}/{sequence_dir.name}"
                annotations = {
                    item.stem: str(item)
                    for item in sorted(sequence_dir.iterdir())
                    if item.is_file() and item.suffix.lower() == ".txt"
                }
                video_path = self._first_file(sequence_dir, ".avi")
                label_path = self._first_mask_path(sequence_dir)

                samples.append(
                    DatasetSample(
                        dataset_name=self.dataset_name,
                        split=split,
                        sample_id=sample_id,
                        sequence_id=sample_id,
                        modality="video_sequence",
                        label_path=str(label_path) if label_path else None,
                        video_path=str(video_path) if video_path else None,
                        annotations=annotations,
                        metadata={
                            "group": group_dir.name,
                            "task": task,
                            "domain": domain,
                            "sequence_name": sequence_dir.name,
                        },
                    )
                )
        return samples

    def _resolve_nested_root(self, group_dir: Path) -> Path | None:
        nested_dirs = sorted(path for path in group_dir.iterdir() if path.is_dir())
        if not nested_dirs:
            return None
        if len(nested_dirs) == 1 and nested_dirs[0].name == group_dir.name:
            nested_group = nested_dirs[0]
            second_level = [path for path in nested_group.iterdir() if path.is_dir()]
            if second_level:
                return nested_group / "Training" if (nested_group / "Training").exists() else nested_group
        return group_dir

    def _infer_split(self, group_name: str) -> str:
        if "Training" in group_name:
            return "train"
        if "Testing" in group_name:
            return "test"
        if "Revision" in group_name:
            return "revision"
        return "unknown"

    def _first_file(self, directory: Path, suffix: str) -> Path | None:
        return next((path for path in sorted(directory.iterdir()) if path.is_file() and path.suffix.lower() == suffix), None)

    def _first_mask_path(self, directory: Path) -> Path | None:
        return next(
            (
                path
                for path in sorted(directory.iterdir())
                if path.is_file() and path.suffix.lower() == ".png" and "class" in path.stem.lower()
            ),
            None,
        )
