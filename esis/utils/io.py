from __future__ import annotations

from pathlib import Path
import base64

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def read_first_video_frame(path: str | Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise FileNotFoundError(f"Failed to read first video frame: {path}")
        return frame
    finally:
        capture.release()


def read_video_frame(path: str | Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        if frame_index > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise FileNotFoundError(f"Failed to read frame {frame_index} from video: {path}")
        return frame
    finally:
        capture.release()


def get_video_frame_count(path: str | Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return max(count, 0)
    finally:
        capture.release()


def write_image(path: str | Path, image: np.ndarray) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    if not cv2.imwrite(str(target), image):
        raise OSError(f"Failed to write image: {target}")
    return target


def as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        return as_bgr(mask)
    mask_u8 = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)


def annotate_image(image: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = as_bgr(image)
    y = 24
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        y += 24
    return canvas


def resize_to_fit(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= max_width and height <= max_height:
        return image.copy()
    scale = min(max_width / width, max_height / height)
    resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    return resized


def image_to_png_base64(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError("Failed to encode image to PNG.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")
