from __future__ import annotations

from typing import Any

from esis.segmentation.adapter_vit_cnn import AdapterVitCnnSegmenter
from esis.segmentation.base import BaseSegmenter
from esis.segmentation.classical import MaskLoaderSegmenter
from esis.segmentation.matis import MatisSegmenter
from esis.segmentation.sam2_zero_shot import Sam2ZeroShotSegmenter
from esis.segmentation.surgsam2 import SurgSam2Segmenter

SEGMENTER_REGISTRY: dict[str, type[BaseSegmenter]] = {
    "mask_loader": MaskLoaderSegmenter,
    "adapter_vit_cnn": AdapterVitCnnSegmenter,
    "matis": MatisSegmenter,
    "surgsam2": SurgSam2Segmenter,
    "sam2_zero_shot": Sam2ZeroShotSegmenter,
}


def available_segmenters() -> list[str]:
    return sorted(SEGMENTER_REGISTRY)


def create_segmenter(name: str, **kwargs: Any) -> BaseSegmenter:
    try:
        segmenter_class = SEGMENTER_REGISTRY[name]
    except KeyError as exc:
        supported = ", ".join(sorted(SEGMENTER_REGISTRY))
        raise ValueError(f"Unknown segmenter '{name}'. Supported values: {supported}") from exc
    return segmenter_class(**kwargs)
