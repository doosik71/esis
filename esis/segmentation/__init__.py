"""Segmentation modules for ESIS."""

from esis.segmentation.base import BaseSegmenter, SegmentationModel, SegmentationResult
from esis.segmentation.classical import ClassicalInstrumentSegmenter, ClassicalSegmentationConfig, MaskLoaderSegmenter
from esis.segmentation.model_wrapper import ModelWrapperConfig, ModelWrapperSegmenter
from esis.segmentation.postprocessing import (
    binary_mask,
    fill_small_holes,
    keep_largest_component,
    morphological_close,
    remove_small_components,
)
from esis.segmentation.preprocessing import (
    ensure_grayscale,
    ensure_uint8,
    normalize_image,
    prepare_model_input,
    resize_like,
    standardize_image,
)

__all__ = [
    "BaseSegmenter",
    "ClassicalInstrumentSegmenter",
    "ClassicalSegmentationConfig",
    "MaskLoaderSegmenter",
    "ModelWrapperConfig",
    "ModelWrapperSegmenter",
    "SegmentationModel",
    "SegmentationResult",
    "binary_mask",
    "ensure_grayscale",
    "ensure_uint8",
    "fill_small_holes",
    "keep_largest_component",
    "morphological_close",
    "normalize_image",
    "prepare_model_input",
    "remove_small_components",
    "resize_like",
    "standardize_image",
]
