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
