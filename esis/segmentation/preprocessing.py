from __future__ import annotations

import cv2
import numpy as np


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.copy()
    clipped = np.clip(image, 0, 255)
    return clipped.astype(np.uint8)


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0].copy()
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_image(image: np.ndarray) -> np.ndarray:
    array = image.astype(np.float32)
    min_value = float(array.min()) if array.size else 0.0
    max_value = float(array.max()) if array.size else 0.0
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def standardize_image(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    array = image.astype(np.float32)
    mean = float(array.mean()) if array.size else 0.0
    std = float(array.std()) if array.size else 0.0
    return (array - mean) / max(std, eps)


def resize_like(image: np.ndarray, target_shape_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape_hw
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def prepare_model_input(image: np.ndarray) -> np.ndarray:
    normalized = normalize_image(image)
    if normalized.ndim == 2:
        normalized = normalized[:, :, None]
    return np.transpose(normalized, (2, 0, 1))
