from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from esis.datasets.schema import DatasetSample
from esis.segmentation.base import BaseSegmenter, SegmentationModel, SegmentationResult
from esis.segmentation.postprocessing import binary_mask, keep_largest_component, remove_small_components
from esis.segmentation.preprocessing import ensure_grayscale, normalize_image


@dataclass(slots=True)
class ModelWrapperConfig:
    apply_sigmoid_threshold: bool = True
    threshold: float = 0.5
    keep_largest_component_only: bool = False
    min_component_area: int = 0


class ModelWrapperSegmenter(BaseSegmenter):
    name = "model_wrapper"

    def __init__(
        self,
        model: SegmentationModel,
        config: ModelWrapperConfig | None = None,
        model_name: str | None = None,
    ) -> None:
        self.model = model
        self.config = config or ModelWrapperConfig()
        self.model_name = model_name or model.__class__.__name__

    def segment(self, image: np.ndarray, sample: DatasetSample | None = None) -> SegmentationResult:
        raw_output = self.model.predict(image, sample=sample)
        mask = self._normalize_model_output(raw_output)

        if self.config.min_component_area > 0:
            mask = remove_small_components(mask, min_area=self.config.min_component_area)
        if self.config.keep_largest_component_only:
            mask = keep_largest_component(mask)

        sample_id = sample.sample_id if sample is not None else "unknown"
        return SegmentationResult(
            sample_id=sample_id,
            mask=mask,
            metadata={
                "segmenter": self.name,
                "model_name": self.model_name,
                "config": {
                    "apply_sigmoid_threshold": self.config.apply_sigmoid_threshold,
                    "threshold": self.config.threshold,
                    "keep_largest_component_only": self.config.keep_largest_component_only,
                    "min_component_area": self.config.min_component_area,
                },
            },
        )

    def _normalize_model_output(self, output: np.ndarray) -> np.ndarray:
        array = np.asarray(output)
        if array.ndim == 3:
            array = ensure_grayscale(array)
        array = normalize_image(array.astype(np.float32))
        if self.config.apply_sigmoid_threshold:
            return binary_mask(array, threshold=self.config.threshold)
        return binary_mask(array)
