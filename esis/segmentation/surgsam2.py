from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import uuid

import numpy as np
from PIL import Image

from esis.datasets.schema import DatasetSample
from esis.segmentation.base import BaseSegmenter, SegmentationResult
from esis.segmentation.postprocessing import postprocess_binary_mask
from esis.segmentation.torch_utils import resolve_device, to_rgb_uint8, vendor_path
from esis.utils.config import project_root, runs_root
from esis.utils.io import ensure_dir


DEFAULT_SURGSAM2_CHECKPOINT_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1DyrrLKst1ZQwkgKM7BWCCwLxSXAgOcMI&export=download&authuser=0"
)


@dataclass(slots=True)
class SurgSam2Config:
    vendor_root: str = "temp/cache/vendors/Surgical-SAM-2"
    model_id: str = "facebook/sam2.1-hiera-tiny"
    checkpoint_path: str = "temp/model/sam2.1_hiera_s_endo18.pth"
    model_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
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
        root = project_root()
        vendor_root = root / self.config.vendor_root
        if not vendor_root.exists():
            raise FileNotFoundError(f"SurgSAM-2 vendor repository not found: {vendor_root}")

        checkpoint_path = root / self.config.checkpoint_path
        if not checkpoint_path.exists():
            print(
                "SurgSAM-2 checkpoint not found at "
                f"{checkpoint_path}. Download it from {DEFAULT_SURGSAM2_CHECKPOINT_URL}"
            )
            raise FileNotFoundError(f"SurgSAM-2 checkpoint not found: {checkpoint_path}")

        with vendor_path(vendor_root):
            from sam2.build_sam import build_sam2_video_predictor

            self.predictor = build_sam2_video_predictor(
                config_file=self.config.model_config,
                ckpt_path=str(checkpoint_path),
                device=str(self.device),
            )

        self.checkpoint_loaded = True
        self.checkpoint_path = str(checkpoint_path)

    def _build_prompts(self, rgb: np.ndarray, sample: DatasetSample | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = rgb.shape[:2]
        points = np.array([[width / 2.0, height / 2.0]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        box = np.array([width * 0.2, height * 0.2, width * 0.8, height * 0.8], dtype=np.float32)
        return points, labels, box

    def segment(self, image: np.ndarray, sample: DatasetSample | None = None) -> SegmentationResult:
        rgb = to_rgb_uint8(image)
        points, labels, box = self._build_prompts(rgb, sample=sample)
        best_mask = self._segment_single_frame(rgb, points=points, labels=labels, box=box)

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
                "checkpoint_path": self.checkpoint_path,
                "device": str(self.device),
                "used_temporal_memory": False,
            },
        )

    def _segment_single_frame(
        self,
        rgb: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        box: np.ndarray,
    ) -> np.ndarray:
        # The SurgSAM-2 checkpoint is trained for the video predictor path.
        # To support the GUI's per-frame API, wrap one frame as a 1-frame sequence.
        temp_dir = ensure_dir(runs_root(project_root()) / "_surgsam2_frames" / uuid.uuid4().hex)
        try:
            frame_path = temp_dir / "00000.jpg"
            Image.fromarray(rgb).save(frame_path)
            inference_state = self.predictor.init_state(video_path=str(temp_dir))
            _, _, video_res_masks = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
                box=box,
            )
            return video_res_masks[0, 0].detach().cpu().numpy()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
