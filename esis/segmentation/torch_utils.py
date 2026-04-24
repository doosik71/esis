from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Iterator

import cv2
import numpy as np
import torch


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def resolve_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare_torch_image(
    image: np.ndarray,
    size_hw: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int]]:
    rgb = to_rgb_uint8(image)
    original_shape = rgb.shape[:2]
    resized = cv2.resize(rgb, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = tensor.to(device)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (tensor - mean) / std, original_shape


def resize_mask_to_original(mask: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    height, width = original_shape
    resized = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return resized


def pick_best_mask(candidates: np.ndarray, scores: np.ndarray | None = None) -> np.ndarray:
    masks = np.asarray(candidates)
    if masks.ndim == 2:
        return masks
    if masks.ndim == 4:
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
    if masks.ndim != 3 or masks.shape[0] == 0:
        raise ValueError("Expected a stack of mask candidates.")

    if scores is None:
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        best_index = int(np.argmax(areas))
        return masks[best_index]

    flat_scores = np.asarray(scores).reshape(-1)
    best_index = int(np.argmax(flat_scores[: masks.shape[0]]))
    return masks[best_index]


def load_state_dict_flexible(module: torch.nn.Module, checkpoint_path: str | None) -> bool:
    if not checkpoint_path:
        return False
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break

    if isinstance(state_dict, dict):
        cleaned = {}
        for key, value in state_dict.items():
            cleaned[key[7:]] = value if not key.startswith("module.") else value
        if any(key.startswith("module.") for key in state_dict):
            cleaned = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
        else:
            cleaned = state_dict
        missing, unexpected = module.load_state_dict(cleaned, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
        return True

    raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")


@contextmanager
def vendor_path(path: str | Path) -> Iterator[None]:
    vendor = str(Path(path).resolve())
    sys.path.insert(0, vendor)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == vendor:
            sys.path.pop(0)
