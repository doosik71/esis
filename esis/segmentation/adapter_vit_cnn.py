from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from esis.datasets.schema import DatasetSample
from esis.segmentation.base import SegmentationModel
from esis.segmentation.model_wrapper import ModelWrapperConfig, ModelWrapperSegmenter
from esis.segmentation.postprocessing import binary_mask
from esis.segmentation.preprocessing import (
    ensure_grayscale,
    ensure_three_channels,
    normalize_image,
    prepare_model_input,
    resize_image,
)


@dataclass(slots=True)
class AdapterVitCnnConfig:
    input_size: tuple[int, int] = (384, 384)
    threshold: float = 0.5
    min_component_area: int = 256
    keep_largest_component_only: bool = False
    checkpoint_path: str | None = None
    use_proxy_if_unavailable: bool = True


class _AdapterVitCnnProxyModel(SegmentationModel):
    def __init__(self, config: AdapterVitCnnConfig) -> None:
        self.config = config

    def predict(self, image: np.ndarray, sample: DatasetSample | None = None) -> np.ndarray:
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))
        resized = resize_image(image, self.config.input_size)
        rgb = ensure_three_channels(resized)
        gray = ensure_grayscale(rgb)

        normalized = normalize_image(gray)
        blurred = cv2.GaussianBlur(normalized, (0, 0), sigmaX=2.0)
        edges = cv2.Canny((normalized * 255).astype(np.uint8), 32, 128).astype(np.float32) / 255.0
        local = cv2.adaptiveThreshold(
            (blurred * 255).astype(np.uint8),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            4,
        ).astype(np.float32) / 255.0
        logits = np.clip(0.55 * local + 0.45 * edges, 0.0, 1.0)
        return logits


class AdapterVitCnnSegmenter(ModelWrapperSegmenter):
    name = "adapter_vit_cnn"

    def __init__(
        self,
        model: SegmentationModel | None = None,
        config: AdapterVitCnnConfig | None = None,
    ) -> None:
        self.backend_config = config or AdapterVitCnnConfig()
        backend_model = model or _AdapterVitCnnProxyModel(self.backend_config)
        super().__init__(
            model=backend_model,
            config=ModelWrapperConfig(
                apply_sigmoid_threshold=True,
                threshold=self.backend_config.threshold,
                keep_largest_component_only=self.backend_config.keep_largest_component_only,
                min_component_area=self.backend_config.min_component_area,
                close_kernel_size=5,
                fill_holes=True,
            ),
            model_name="adapter_vit_cnn",
            backend_name=self.name,
            preprocess=self._preprocess,
            postprocess=self._postprocess,
        )

    def _preprocess(self, image: np.ndarray, sample: DatasetSample | None = None) -> dict[str, Any]:
        resized = resize_image(image, self.backend_config.input_size)
        chw = prepare_model_input(ensure_three_channels(resized)).astype(np.float32)
        return {"network_input": chw, "resized_image": resized, "original_shape": image.shape[:2]}

    def _postprocess(
        self,
        output: Any,
        original_image: np.ndarray,
        sample: DatasetSample | None = None,
        prepared: Any | None = None,
    ) -> np.ndarray:
        if isinstance(output, dict):
            array = np.asarray(output.get("mask"))
        else:
            array = np.asarray(output)
        if array.ndim == 3:
            array = array[0]
        mask = binary_mask(array, threshold=self.backend_config.threshold)
        return resize_image(mask, original_image.shape[:2], interpolation=cv2.INTER_NEAREST)
