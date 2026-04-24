from __future__ import annotations

import cv2
import numpy as np


def binary_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    array = np.asarray(mask)
    if array.dtype != np.uint8:
        array = (array >= threshold).astype(np.uint8) * 255
    else:
        array = (array > 0).astype(np.uint8) * 255
    return array


def sigmoid(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-array))


def softmax(values: np.ndarray, axis: int = 0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def morphological_close(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size <= 1:
        return binary_mask(mask)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(binary_mask(mask), cv2.MORPH_CLOSE, kernel)


def fill_small_holes(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return morphological_close(mask, kernel_size=kernel_size)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return binary_mask(mask)
    src = binary_mask(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    filtered = np.zeros_like(src)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == label] = 255
    return filtered


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    src = binary_mask(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    if num_labels <= 1:
        return src
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    output = np.zeros_like(src)
    output[labels == largest_label] = 255
    return output


def resize_mask(mask: np.ndarray, target_shape_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape_hw
    return cv2.resize(binary_mask(mask), (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def postprocess_binary_mask(
    mask: np.ndarray,
    min_component_area: int = 0,
    keep_largest_component_only: bool = False,
    close_kernel_size: int = 0,
    fill_holes: bool = False,
) -> np.ndarray:
    refined = binary_mask(mask)
    if close_kernel_size > 1:
        refined = morphological_close(refined, kernel_size=close_kernel_size)
    if fill_holes:
        refined = fill_small_holes(refined, kernel_size=max(close_kernel_size, 3))
    if min_component_area > 0:
        refined = remove_small_components(refined, min_area=min_component_area)
    if keep_largest_component_only:
        refined = keep_largest_component(refined)
    return refined
