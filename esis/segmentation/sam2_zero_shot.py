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
class Sam2ZeroShotConfig:
    input_size: tuple[int, int] = (512, 512)
    threshold: float = 0.55
    min_component_area: int = 160
    checkpoint_path: str | None = None
    center_prior_weight: float = 0.2


class _Sam2ZeroShotProxyModel(SegmentationModel):
    def __init__(self, config: Sam2ZeroShotConfig) -> None:
        self.config = config

    def predict(self, image: np.ndarray, sample: DatasetSample | None = None) -> np.ndarray:
        resized = resize_image(image, self.config.input_size)
        gray = ensure_grayscale(resized).astype(np.float32) / 255.0

        gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_energy = cv2.magnitude(gradient_x, gradient_y)
        edge_energy = edge_energy / max(float(edge_energy.max()), 1e-6)

        height, width = gray.shape[:2]
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        cy = (height - 1) / 2.0
        cx = (width - 1) / 2.0
        radius = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        center_prior = 1.0 - (radius / max(float(radius.max()), 1e-6))
        logits = np.clip((1.0 - self.config.center_prior_weight) * edge_energy + self.config.center_prior_weight * center_prior, 0.0, 1.0)
        return logits


class Sam2ZeroShotSegmenter(ModelWrapperSegmenter):
    name = "sam2_zero_shot"

    def __init__(
        self,
        model: SegmentationModel | None = None,
        config: Sam2ZeroShotConfig | None = None,
    ) -> None:
        self.backend_config = config or Sam2ZeroShotConfig()
        backend_model = model or _Sam2ZeroShotProxyModel(self.backend_config)
        super().__init__(
            model=backend_model,
            config=ModelWrapperConfig(
                apply_sigmoid_threshold=True,
                threshold=self.backend_config.threshold,
                keep_largest_component_only=True,
                min_component_area=self.backend_config.min_component_area,
                close_kernel_size=5,
                fill_holes=True,
            ),
            model_name="sam2_zero_shot",
            backend_name=self.name,
            preprocess=self._preprocess,
            postprocess=self._postprocess,
        )

    def _preprocess(self, image: np.ndarray, sample: DatasetSample | None = None) -> dict[str, Any]:
        return {
            "resized_image": resize_image(image, self.backend_config.input_size),
            "original_shape": image.shape[:2],
            "sample_id": sample.sample_id if sample is not None else "unknown",
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
