from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor

from esis.datasets.schema import DatasetSample
from esis.segmentation.base import BaseSegmenter, SegmentationResult
from esis.segmentation.postprocessing import postprocess_binary_mask
from esis.segmentation.torch_utils import pick_best_mask, resolve_device, to_rgb_uint8


@dataclass(slots=True)
class Sam2ZeroShotConfig:
    model_id: str = "facebook/sam2.1-hiera-tiny"
    device: str | None = None
    threshold: float = 0.0
    min_component_area: int = 96
    keep_largest_component_only: bool = True
    close_kernel_size: int = 5


class Sam2ZeroShotSegmenter(BaseSegmenter):
    name = "sam2_zero_shot"

    def __init__(self, config: Sam2ZeroShotConfig | None = None) -> None:
        self.config = config or Sam2ZeroShotConfig()
        self.device = resolve_device(self.config.device)
        self.processor = self._load_processor()
        self.model = self._load_model().to(self.device)
        self.model.eval()
        self.checkpoint_loaded = True

    def _load_processor(self) -> Sam2Processor:
        try:
            return Sam2Processor.from_pretrained(self.config.model_id, local_files_only=True)
        except OSError:
            return Sam2Processor.from_pretrained(self.config.model_id)

    def _load_model(self) -> Sam2Model:
        try:
            return Sam2Model.from_pretrained(self.config.model_id, local_files_only=True)
        except OSError:
            return Sam2Model.from_pretrained(self.config.model_id)

    @torch.inference_mode()
    def segment(self, image: np.ndarray, sample: DatasetSample | None = None) -> SegmentationResult:
        rgb = to_rgb_uint8(image)
        height, width = rgb.shape[:2]
        center_point = [[[[width / 2.0, height / 2.0]]]]
        center_label = [[[1]]]
        prompt_box = [[[width * 0.2, height * 0.2, width * 0.8, height * 0.8]]]

        inputs = self.processor(
            images=rgb,
            input_points=center_point,
            input_labels=center_label,
            input_boxes=prompt_box,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs, multimask_output=True)
        processed_masks = self.processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            mask_threshold=self.config.threshold,
            binarize=False,
        )
        scores = outputs.iou_scores.detach().cpu().numpy()
        candidates = processed_masks[0].numpy()
        best_mask = pick_best_mask(candidates, scores=scores)
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
                "model_type": "transformers_sam2",
                "model_id": self.config.model_id,
                "device": str(self.device),
            },
        )
