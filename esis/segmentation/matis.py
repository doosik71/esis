from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from esis.datasets.schema import DatasetSample
from esis.segmentation.base import SegmentationModel
from esis.segmentation.model_wrapper import ModelWrapperConfig, ModelWrapperSegmenter
from esis.segmentation.postprocessing import binary_mask
from esis.segmentation.preprocessing import ensure_grayscale, resize_image


@dataclass(slots=True)
class MatisConfig:
    input_size: tuple[int, int] = (448, 448)
    threshold: float = 0.48
    min_component_area: int = 192
    keep_largest_component_only: bool = False
    checkpoint_path: str | None = None
    temporal_blend: float = 0.15


class _MatisProxyModel(SegmentationModel):
    def __init__(self, config: MatisConfig) -> None:
        self.config = config

    def predict(self, image: np.ndarray, sample: DatasetSample | None = None) -> np.ndarray:
        resized = resize_image(image, self.config.input_size)
        gray = ensure_grayscale(resized)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        laplacian = cv2.convertScaleAbs(laplacian).astype(np.float32) / 255.0

        _, coarse = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        coarse = coarse.astype(np.float32) / 255.0

        attention = cv2.GaussianBlur(coarse, (0, 0), 6.0)
        logits = np.clip(0.7 * attention + 0.3 * laplacian, 0.0, 1.0)
        return logits


class MatisSegmenter(ModelWrapperSegmenter):
    name = "matis"

    def __init__(
        self,
        model: SegmentationModel | None = None,
        config: MatisConfig | None = None,
    ) -> None:
        self.backend_config = config or MatisConfig()
        backend_model = model or _MatisProxyModel(self.backend_config)
        super().__init__(
            model=backend_model,
            config=ModelWrapperConfig(
                apply_sigmoid_threshold=True,
                threshold=self.backend_config.threshold,
                keep_largest_component_only=self.backend_config.keep_largest_component_only,
                min_component_area=self.backend_config.min_component_area,
                close_kernel_size=7,
                fill_holes=True,
            ),
            model_name="matis",
            backend_name=self.name,
            preprocess=self._preprocess,
            postprocess=self._postprocess,
        )

    def _preprocess(self, image: np.ndarray, sample: DatasetSample | None = None) -> dict[str, Any]:
        return {
            "resized_image": resize_image(image, self.backend_config.input_size),
            "original_shape": image.shape[:2],
        }

    def _postprocess(
        self,
        output: Any,
        original_image: np.ndarray,
        sample: DatasetSample | None = None,
        prepared: Any | None = None,
    ) -> np.ndarray:
        array = np.asarray(output)
        if array.ndim == 3:
            array = array[0]
        mask = binary_mask(array, threshold=self.backend_config.threshold)
        return resize_image(mask, original_image.shape[:2], interpolation=cv2.INTER_NEAREST)
