from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np

from esis.datasets.schema import DatasetSample


@dataclass(slots=True)
class SegmentationResult:
    sample_id: str
    mask: np.ndarray
    score: float | None = None
    label_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mask_shape"] = list(self.mask.shape)
        payload["mask_dtype"] = str(self.mask.dtype)
        payload.pop("mask")
        return payload


class SegmentationModel(Protocol):
    def predict(self, image: Any, sample: DatasetSample | None = None) -> Any:
        """Return a mask-like output for one image."""


class BaseSegmenter(ABC):
    name: str = "base"

    @abstractmethod
    def segment(self, image: np.ndarray, sample: DatasetSample | None = None) -> SegmentationResult:
        """Return a segmentation result for one image."""

    def segment_batch(
        self,
        images: list[np.ndarray],
        samples: list[DatasetSample] | None = None,
    ) -> list[SegmentationResult]:
        results: list[SegmentationResult] = []
        for index, image in enumerate(images):
            sample = samples[index] if samples is not None and index < len(samples) else None
            results.append(self.segment(image, sample=sample))
        return results

    def get_name(self) -> str:
        return self.name
