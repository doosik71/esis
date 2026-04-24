from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from esis.datasets.schema import DatasetSample
from esis.segmentation.base import BaseSegmenter, SegmentationModel, SegmentationResult
from esis.segmentation.postprocessing import (
    binary_mask,
    keep_largest_component,
    postprocess_binary_mask,
    remove_small_components,
)
from esis.segmentation.preprocessing import ensure_grayscale, normalize_image

PreprocessHook = Callable[[np.ndarray, DatasetSample | None], Any]
PostprocessHook = Callable[[Any, np.ndarray, DatasetSample | None, Any | None], np.ndarray]


@dataclass(slots=True)
class ModelWrapperConfig:
    apply_sigmoid_threshold: bool = True
    threshold: float = 0.5
    keep_largest_component_only: bool = False
    min_component_area: int = 0
    close_kernel_size: int = 0
    fill_holes: bool = False


class ModelWrapperSegmenter(BaseSegmenter):
    name = "model_wrapper"

    def __init__(
        self,
        model: SegmentationModel,
        config: ModelWrapperConfig | None = None,
        model_name: str | None = None,
        backend_name: str | None = None,
        preprocess: PreprocessHook | None = None,
        postprocess: PostprocessHook | None = None,
    ) -> None:
        self.model = model
        self.config = config or ModelWrapperConfig()
        self.model_name = model_name or model.__class__.__name__
        self.name = backend_name or self.name
        self._preprocess_hook = preprocess
        self._postprocess_hook = postprocess

    def segment(self, image: np.ndarray, sample: DatasetSample | None = None) -> SegmentationResult:
        prepared = self._prepare_model_input(image, sample=sample)
        model_input = self._extract_model_input(prepared, image)
        raw_output = self.model.predict(model_input, sample=sample)
        mask = self._normalize_model_output(raw_output, image=image, sample=sample, prepared=prepared)

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
                    "close_kernel_size": self.config.close_kernel_size,
                    "fill_holes": self.config.fill_holes,
                },
            },
        )

    def _prepare_model_input(self, image: np.ndarray, sample: DatasetSample | None = None) -> Any:
        if self._preprocess_hook is None:
            return image
        return self._preprocess_hook(image, sample)

    def _extract_model_input(self, prepared: Any, image: np.ndarray) -> Any:
        if isinstance(prepared, dict) and "network_input" in prepared:
            return prepared["network_input"]
        if isinstance(prepared, dict) and "resized_image" in prepared:
            return prepared["resized_image"]
        return prepared

    def _normalize_model_output(
        self,
        output: np.ndarray,
        image: np.ndarray,
        sample: DatasetSample | None = None,
        prepared: Any | None = None,
    ) -> np.ndarray:
        if self._postprocess_hook is not None:
            mask = self._postprocess_hook(output, image, sample, prepared)
        else:
            mask = self._default_postprocess(output)

        return postprocess_binary_mask(
            mask,
            min_component_area=self.config.min_component_area,
            keep_largest_component_only=self.config.keep_largest_component_only,
            close_kernel_size=self.config.close_kernel_size,
            fill_holes=self.config.fill_holes,
        )

    def _default_postprocess(self, output: np.ndarray) -> np.ndarray:
        array = np.asarray(output)
        if array.ndim == 3:
            array = ensure_grayscale(array)
        array = normalize_image(array.astype(np.float32))
        if self.config.apply_sigmoid_threshold:
            return binary_mask(array, threshold=self.config.threshold)
        return binary_mask(array)
