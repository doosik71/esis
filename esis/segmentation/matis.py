from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import timm
import torch
from torch import nn

from esis.datasets.schema import DatasetSample
from esis.segmentation.base import BaseSegmenter, SegmentationResult
from esis.segmentation.checkpoints import resolve_matis_checkpoint
from esis.segmentation.postprocessing import postprocess_binary_mask
from esis.segmentation.torch_utils import load_state_dict_flexible, prepare_torch_image, resolve_device


class _MaskedAttentionDecoder(nn.Module):
    def __init__(self, channels: list[int], decoder_dim: int, num_queries: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(channel, decoder_dim, kernel_size=1) for channel in channels])
        layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, decoder_dim))
        self.class_head = nn.Linear(decoder_dim, 2)
        self.mask_head = nn.Linear(decoder_dim, decoder_dim)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target_hw = features[0].shape[-2:]
        fused = None
        for feature, projection in zip(features, self.projections):
            mapped = projection(feature)
            if mapped.shape[-2:] != target_hw:
                mapped = torch.nn.functional.interpolate(
                    mapped,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            fused = mapped if fused is None else fused + mapped

        batch_size, channels, height, width = fused.shape
        memory = fused.flatten(2).transpose(1, 2)
        queries = self.query_embed.expand(batch_size, -1, -1)
        decoded = self.decoder(queries, memory)

        class_logits = self.class_head(decoded)
        class_prob = torch.softmax(class_logits, dim=-1)[..., 1]
        mask_embed = self.mask_head(decoded)
        pixel_embed = fused.flatten(2)
        query_masks = torch.einsum("bqc,bch->bqh", mask_embed, pixel_embed).view(batch_size, -1, height, width)
        weighted_masks = query_masks * class_prob[:, :, None, None]
        return weighted_masks.max(dim=1, keepdim=True).values


class MatisNet(nn.Module):
    def __init__(self, backbone_name: str, pretrained_backbone: bool, decoder_dim: int, num_queries: int) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained_backbone, features_only=True, out_indices=(0, 1, 2, 3))
        channels = list(self.backbone.feature_info.channels())
        self.decoder = _MaskedAttentionDecoder(
            channels=channels,
            decoder_dim=decoder_dim,
            num_queries=num_queries,
            num_heads=8,
            num_layers=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.decoder(list(features))


@dataclass(slots=True)
class MatisConfig:
    backbone_name: str = "sam2_hiera_tiny"
    pretrained_backbone: bool = True
    input_size: tuple[int, int] = (512, 512)
    decoder_dim: int = 128
    num_queries: int = 8
    checkpoint_path: str | None = None
    dataset_name: str | None = None
    fold: int | None = None
    device: str | None = None
    threshold: float = 0.45
    min_component_area: int = 96
    keep_largest_component_only: bool = True
    close_kernel_size: int = 5


class MatisSegmenter(BaseSegmenter):
    name = "matis"

    def __init__(self, config: MatisConfig | None = None) -> None:
        self.config = config or MatisConfig()
        self.device = resolve_device(self.config.device)
        self.checkpoint_resolution = resolve_matis_checkpoint(
            dataset_name=self.config.dataset_name,
            explicit_path=self.config.checkpoint_path,
            fold=self.config.fold,
        )
        self.model = MatisNet(
            backbone_name=self.config.backbone_name,
            pretrained_backbone=self.config.pretrained_backbone,
            decoder_dim=self.config.decoder_dim,
            num_queries=self.config.num_queries,
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
        fold = self.config.fold
        if fold is None and sample.dataset_name == "endovis17":
            fold = 3
        resolution = resolve_matis_checkpoint(
            dataset_name=sample.dataset_name,
            explicit_path=self.config.checkpoint_path,
            fold=fold,
        )
        if resolution.checkpoint_path is None:
            self.checkpoint_resolution = resolution
            self.config.dataset_name = sample.dataset_name
            self.config.fold = fold
            return
        self.checkpoint_loaded = load_state_dict_flexible(self.model, resolution.checkpoint_path)
        self.checkpoint_resolution = resolution
        self.config.dataset_name = sample.dataset_name
        self.config.fold = fold
