from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from esis.datasets.schema import DatasetSample
from esis.segmentation.base import BaseSegmenter, SegmentationResult
from esis.segmentation.preprocessing import (
    ensure_grayscale,
    ensure_uint8,
    normalize_image,
    resize_like,
)
from esis.segmentation.postprocessing import (
    binary_mask,
    fill_small_holes,
    keep_largest_component,
    morphological_close,
    remove_small_components,
)
from esis.utils.io import read_image


@dataclass(slots=True)
class ClassicalSegmentationConfig:
    blur_kernel_size: int = 5
    threshold_value: int = 0
    min_component_area: int = 512
    close_kernel_size: int = 7
    use_otsu: bool = True


class MaskLoaderSegmenter(BaseSegmenter):
    name = "mask_loader"

    def segment(self, image: np.ndarray, sample: DatasetSample | None = None) -> SegmentationResult:
        if sample is None or not sample.label_path:
            raise ValueError("MaskLoaderSegmenter requires a dataset sample with label_path.")

        label = read_image(sample.label_path)
        mask = ensure_grayscale(label)
        if mask.shape[:2] != image.shape[:2]:
            mask = resize_like(mask, image.shape[:2])
        mask = binary_mask(mask)
        return SegmentationResult(
            sample_id=sample.sample_id,
            mask=mask,
            score=1.0,
            label_path=sample.label_path,
            metadata={"segmenter": self.name, "source": "dataset_label"},
        )


class ClassicalInstrumentSegmenter(BaseSegmenter):
    name = "classical_threshold"

    def __init__(self, config: ClassicalSegmentationConfig | None = None) -> None:
        self.config = config or ClassicalSegmentationConfig()

    def segment(self, image: np.ndarray, sample: DatasetSample | None = None) -> SegmentationResult:
        gray = ensure_grayscale(image)
        gray = normalize_image(gray)
        gray = ensure_uint8(gray)

        if self.config.blur_kernel_size > 1:
            gray = cv2.GaussianBlur(gray, (self.config.blur_kernel_size, self.config.blur_kernel_size), 0)

        threshold_mode = cv2.THRESH_BINARY_INV
        if self.config.use_otsu:
            _, mask = cv2.threshold(gray, 0, 255, threshold_mode | cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(gray, self.config.threshold_value, 255, threshold_mode)

        mask = binary_mask(mask)
        mask = morphological_close(mask, self.config.close_kernel_size)
        mask = fill_small_holes(mask, self.config.close_kernel_size)
        mask = remove_small_components(mask, min_area=self.config.min_component_area)
        mask = keep_largest_component(mask)

        sample_id = sample.sample_id if sample is not None else "unknown"
        return SegmentationResult(
            sample_id=sample_id,
            mask=mask,
            metadata={
                "segmenter": self.name,
                "config": {
                    "blur_kernel_size": self.config.blur_kernel_size,
                    "threshold_value": self.config.threshold_value,
                    "min_component_area": self.config.min_component_area,
                    "close_kernel_size": self.config.close_kernel_size,
                    "use_otsu": self.config.use_otsu,
                },
            },
        )
