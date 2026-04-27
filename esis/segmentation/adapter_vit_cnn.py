from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import timm
import torch
from torch import nn

from esis.datasets.schema import DatasetSample
from esis.segmentation.base import BaseSegmenter, SegmentationResult
from esis.segmentation.checkpoints import CheckpointResolution, resolve_adapter_vit_cnn_checkpoint
from esis.segmentation.postprocessing import postprocess_binary_mask
from esis.segmentation.torch_utils import (
    load_state_dict_flexible,
    prepare_torch_image,
    resolve_device,
)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AdapterVitCnnNet(nn.Module):
    def __init__(self, backbone_name: str, decoder_channels: int, pretrained_backbone: bool) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained_backbone, num_classes=0)
        embed_dim = int(getattr(self.backbone, "embed_dim", 384))
        self.projections = nn.ModuleList([nn.Conv2d(embed_dim, decoder_channels, kernel_size=1) for _ in range(4)])
        self.fuse = _ConvBlock(decoder_channels * 4, decoder_channels)
        self.head = nn.Sequential(
            _ConvBlock(decoder_channels, decoder_channels),
            nn.Conv2d(decoder_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_intermediates(
            x,
            indices=[-4, -3, -2, -1],
            norm=False,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        target_hw = features[0].shape[-2:]
        upsampled = []
        for feature, projection in zip(features, self.projections):
            projected = projection(feature)
            if projected.shape[-2:] != target_hw:
                projected = torch.nn.functional.interpolate(
                    projected,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            upsampled.append(projected)
        fused = self.fuse(torch.cat(upsampled, dim=1))
        logits = self.head(fused)
        return logits


@dataclass(slots=True)
class AdapterVitCnnConfig:
    backbone_name: str = "vit_small_patch14_dinov2"
    pretrained_backbone: bool = True
    input_size: tuple[int, int] = (518, 518)
    decoder_channels: int = 128
    checkpoint_path: str | None = None
    dataset_name: str | None = None
    device: str | None = None
    threshold: float = 0.5
    min_component_area: int = 128
    keep_largest_component_only: bool = True
    close_kernel_size: int = 5


class AdapterVitCnnSegmenter(BaseSegmenter):
    name = "adapter_vit_cnn"

    def __init__(self, config: AdapterVitCnnConfig | None = None) -> None:
        self.config = config or AdapterVitCnnConfig()
        self.device = resolve_device(self.config.device)
        self.checkpoint_resolution = resolve_adapter_vit_cnn_checkpoint(
            dataset_name=self.config.dataset_name,
            explicit_path=self.config.checkpoint_path,
        )
        self.model = AdapterVitCnnNet(
            backbone_name=self.config.backbone_name,
            decoder_channels=self.config.decoder_channels,
            pretrained_backbone=self.config.pretrained_backbone,
        ).to(self.device)
        self.model.eval()
        self.checkpoint_loaded = load_state_dict_flexible(self.model, self.checkpoint_resolution.checkpoint_path)

    @torch.inference_mode()
    def segment(self, image: np.ndarray, sample: DatasetSample | None = None) -> SegmentationResult:
        self._maybe_load_dataset_checkpoint(sample)
        tensor, original_shape = prepare_torch_image(image, self.config.input_size, self.device)
        logits = self.model(tensor)
        logits = torch.nn.functional.interpolate(logits, size=original_shape, mode="bilinear", align_corners=False)
        probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        mask = (probability >= self.config.threshold).astype(np.uint8) * 255
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
                "model_type": "torch",
                "backbone_name": self.config.backbone_name,
                "pretrained_backbone": self.config.pretrained_backbone,
                "checkpoint_loaded": self.checkpoint_loaded,
                "checkpoint_path": self.checkpoint_resolution.checkpoint_path,
                "checkpoint_source": self.checkpoint_resolution.source,
                "device": str(self.device),
            },
        )

    def _maybe_load_dataset_checkpoint(self, sample: DatasetSample | None) -> None:
        if self.checkpoint_loaded:
            return
        if sample is None or not sample.dataset_name:
            return
        if self.config.dataset_name == sample.dataset_name:
            return
        resolution = resolve_adapter_vit_cnn_checkpoint(
            dataset_name=sample.dataset_name,
            explicit_path=self.config.checkpoint_path,
        )
        if resolution.checkpoint_path is None:
            self.checkpoint_resolution = resolution
            self.config.dataset_name = sample.dataset_name
            return
        self.checkpoint_loaded = load_state_dict_flexible(self.model, resolution.checkpoint_path)
        self.checkpoint_resolution = resolution
        self.config.dataset_name = sample.dataset_name
