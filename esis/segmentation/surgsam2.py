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
class SurgSam2Config:
    input_size: tuple[int, int] = (512, 512)
    threshold: float = 0.52
    min_component_area: int = 160
    checkpoint_path: str | None = None
    temporal_smoothing: float = 0.65


class _SurgSam2ProxyModel(SegmentationModel):
    def __init__(self, config: SurgSam2Config) -> None:
        self.config = config
        self.previous_logits_by_sequence: dict[str, np.ndarray] = {}

    def predict(self, image: np.ndarray, sample: DatasetSample | None = None) -> np.ndarray:
        resized = resize_image(image, self.config.input_size)
        gray = ensure_grayscale(resized)
        gray_f = gray.astype(np.float32) / 255.0

        _, coarse = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        coarse = cv2.GaussianBlur(coarse.astype(np.float32) / 255.0, (0, 0), 4.0)

        sequence_id = sample.sequence_id if sample is not None else "__default__"
        if sequence_id in self.previous_logits_by_sequence:
            previous = self.previous_logits_by_sequence[sequence_id]
            logits = self.config.temporal_smoothing * previous + (1.0 - self.config.temporal_smoothing) * coarse
        else:
            logits = coarse
        self.previous_logits_by_sequence[sequence_id] = logits
        return np.clip(logits + 0.15 * gray_f, 0.0, 1.0)


class SurgSam2Segmenter(ModelWrapperSegmenter):
    name = "surgsam2"

    def __init__(
        self,
        model: SegmentationModel | None = None,
        config: SurgSam2Config | None = None,
    ) -> None:
        self.backend_config = config or SurgSam2Config()
        backend_model = model or _SurgSam2ProxyModel(self.backend_config)
        super().__init__(
            model=backend_model,
            config=ModelWrapperConfig(
                apply_sigmoid_threshold=True,
                threshold=self.backend_config.threshold,
                keep_largest_component_only=True,
                min_component_area=self.backend_config.min_component_area,
                close_kernel_size=7,
                fill_holes=True,
            ),
            model_name="surgsam2",
            backend_name=self.name,
            preprocess=self._preprocess,
            postprocess=self._postprocess,
        )

    def _preprocess(self, image: np.ndarray, sample: DatasetSample | None = None) -> dict[str, Any]:
        return {
            "resized_image": resize_image(image, self.backend_config.input_size),
            "original_shape": image.shape[:2],
            "sequence_id": sample.sequence_id if sample is not None else "unknown",
            "frame_index": sample.frame_index if sample is not None else None,
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
