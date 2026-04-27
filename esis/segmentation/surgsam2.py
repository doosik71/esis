from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from esis.datasets.schema import DatasetSample
from esis.segmentation.base import BaseSegmenter, SegmentationResult
from esis.segmentation.postprocessing import postprocess_binary_mask
from esis.segmentation.torch_utils import pick_best_mask, resolve_device, to_rgb_uint8, vendor_path


@dataclass(slots=True)
class SurgSam2Config:
    vendor_root: str = "temp/cache/vendors/Surgical-SAM-2"
    model_id: str = "facebook/sam2.1-hiera-tiny"
    device: str | None = None
    threshold: float = 0.0
    min_component_area: int = 96
    keep_largest_component_only: bool = True
    close_kernel_size: int = 5
    use_temporal_memory: bool = True


class SurgSam2Segmenter(BaseSegmenter):
    name = "surgsam2"

    def __init__(self, config: SurgSam2Config | None = None) -> None:
        self.config = config or SurgSam2Config()
        self.device = resolve_device(self.config.device)
        vendor_root = Path(self.config.vendor_root)
        if not vendor_root.exists():
            raise FileNotFoundError(f"SurgSAM-2 vendor repository not found: {vendor_root}")

        with vendor_path(vendor_root):
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            self.predictor = SAM2ImagePredictor.from_pretrained(self.config.model_id, device=str(self.device))

        self.previous_low_res_masks: dict[str, np.ndarray] = {}
        self.checkpoint_loaded = True

    def _build_prompts(self, rgb: np.ndarray, sample: DatasetSample | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = rgb.shape[:2]
        points = np.array([[width / 2.0, height / 2.0]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        box = np.array([width * 0.2, height * 0.2, width * 0.8, height * 0.8], dtype=np.float32)
        return points, labels, box

    def segment(self, image: np.ndarray, sample: DatasetSample | None = None) -> SegmentationResult:
        rgb = to_rgb_uint8(image)
        sequence_id = sample.sequence_id if sample is not None else "__default__"
        points, labels, box = self._build_prompts(rgb, sample=sample)

        self.predictor.set_image(rgb)
        previous = self.previous_low_res_masks.get(sequence_id) if self.config.use_temporal_memory else None
        masks, scores, low_res = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box,
            mask_input=previous,
            multimask_output=True,
            return_logits=True,
        )
        best_mask = pick_best_mask(masks, scores=scores)
        best_low_res = pick_best_mask(low_res, scores=scores)
        if self.config.use_temporal_memory:
            self.previous_low_res_masks[sequence_id] = best_low_res[None, ...]

        mask = (best_mask > self.config.threshold).astype(np.uint8) * 255
        mask = postprocess_binary_mask(
            mask,
            min_component_area=self.config.min_component_area,
            keep_largest_component_only=self.config.keep_largest_component_only,
            close_kernel_size=self.config.close_kernel_size,
            fill_holes=True,
        )
        sample_id = sample.sample_id if sample is not None else "unknown"
        return SegmentationResult(
            sample_id=sample_id,
            mask=mask,
            metadata={
                "segmenter": self.name,
                "model_type": "official_surgsam2",
                "model_id": self.config.model_id,
                "device": str(self.device),
                "used_temporal_memory": self.config.use_temporal_memory,
            },
        )
