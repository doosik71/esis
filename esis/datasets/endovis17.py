from __future__ import annotations

import re
from pathlib import Path

from esis.datasets.base import BaseDatasetAdapter
from esis.datasets.schema import DatasetAdapterError, DatasetSample

FRAME_PATTERN = re.compile(r"seq_(?P<sequence>\d+)_frame(?P<frame>\d+)", re.IGNORECASE)


class EndoVis17Adapter(BaseDatasetAdapter):
    dataset_name = "endovis17"
    index_file_name = "endovis17_index.json"

    def collect_samples(self) -> list[DatasetSample]:
        root = self.dataset_root / "data" / "endovis2017"
        if not root.exists():
            raise DatasetAdapterError(f"Missing dataset root: {root}")

        samples: list[DatasetSample] = []
        for split_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            image_dir = split_dir / "image"
            label_dir = split_dir / "label"
            if not image_dir.exists():
                continue

            for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
                frame_match = FRAME_PATTERN.search(image_path.stem)
                frame_index = int(frame_match.group("frame")) if frame_match else None
                sequence_id = (
                    f"{split_dir.name}/seq_{frame_match.group('sequence')}"
                    if frame_match
                    else f"{split_dir.name}/{image_path.stem}"
                )
                label_path = label_dir / image_path.name if label_dir.exists() and (label_dir / image_path.name).exists() else None
                samples.append(
                    DatasetSample(
                        dataset_name=self.dataset_name,
                        split=split_dir.name,
                        sample_id=f"{split_dir.name}/{image_path.name}",
                        sequence_id=sequence_id,
                        modality="image_frame",
                        image_path=str(image_path),
                        label_path=str(label_path) if label_path else None,
                        frame_index=frame_index,
                        metadata={
                            "split_group": self._split_group(split_dir.name),
                        },
                    )
                )
        return samples

    def _split_group(self, split_name: str) -> str:
        return "train" if split_name == "train" else "validation"
