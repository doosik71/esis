"""Segmentation modules for ESIS."""

from esis.segmentation.adapter_vit_cnn import AdapterVitCnnConfig, AdapterVitCnnSegmenter
from esis.segmentation.base import BaseSegmenter, SegmentationModel, SegmentationResult
from esis.segmentation.checkpoints import CheckpointResolution, resolve_adapter_vit_cnn_checkpoint, resolve_matis_checkpoint
from esis.segmentation.classical import ClassicalInstrumentSegmenter, ClassicalSegmentationConfig, MaskLoaderSegmenter
from esis.segmentation.factory import SEGMENTER_REGISTRY, available_segmenters, create_segmenter
from esis.segmentation.matis import MatisConfig, MatisSegmenter
from esis.segmentation.model_wrapper import ModelWrapperConfig, ModelWrapperSegmenter
from esis.segmentation.postprocessing import (
    binary_mask,
    fill_small_holes,
    keep_largest_component,
    morphological_close,
    postprocess_binary_mask,
    remove_small_components,
    resize_mask,
    sigmoid,
    softmax,
)
from esis.segmentation.preprocessing import (
    ensure_grayscale,
    ensure_three_channels,
    ensure_uint8,
    imagenet_normalize,
    normalize_image,
    pad_to_shape,
    pad_to_stride,
    prepare_imagenet_input,
    prepare_model_input,
    resize_image,
    resize_like,
    standardize_image,
)
from esis.segmentation.sam2_zero_shot import Sam2ZeroShotConfig, Sam2ZeroShotSegmenter
from esis.segmentation.surgsam2 import SurgSam2Config, SurgSam2Segmenter

__all__ = [
    "AdapterVitCnnConfig",
    "AdapterVitCnnSegmenter",
    "BaseSegmenter",
    "ClassicalInstrumentSegmenter",
    "ClassicalSegmentationConfig",
    "CheckpointResolution",
    "MatisConfig",
    "MatisSegmenter",
    "MaskLoaderSegmenter",
    "ModelWrapperConfig",
    "ModelWrapperSegmenter",
    "SEGMENTER_REGISTRY",
    "Sam2ZeroShotConfig",
    "Sam2ZeroShotSegmenter",
    "SegmentationModel",
    "SegmentationResult",
    "SurgSam2Config",
    "SurgSam2Segmenter",
    "available_segmenters",
    "binary_mask",
    "create_segmenter",
    "ensure_grayscale",
    "ensure_three_channels",
    "ensure_uint8",
    "fill_small_holes",
    "imagenet_normalize",
    "keep_largest_component",
    "morphological_close",
    "normalize_image",
    "pad_to_shape",
    "pad_to_stride",
    "postprocess_binary_mask",
    "prepare_imagenet_input",
    "prepare_model_input",
    "remove_small_components",
    "resolve_adapter_vit_cnn_checkpoint",
    "resolve_matis_checkpoint",
    "resize_image",
    "resize_mask",
    "resize_like",
    "sigmoid",
    "softmax",
    "standardize_image",
]
