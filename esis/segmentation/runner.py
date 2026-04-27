from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from esis.datasets import DatasetSample, ensure_dataset_index
from esis.segmentation import available_segmenters, create_segmenter
from esis.segmentation.base import SegmentationResult
from esis.utils.config import project_root, runs_root
from esis.utils.io import as_bgr, colorize_mask, ensure_dir, read_image, read_video_frame, write_image


@dataclass(slots=True)
class SegmentationRunSelection:
    dataset_name: str
    backend_name: str
    sample_id: str | None = None
    split: str | None = None
    sequence_id: str | None = None
    limit: int = 0


def run_segmentation_selection(
    selection: SegmentationRunSelection,
    root: Path | None = None,
) -> dict[str, Any]:
    proj_root = root or project_root()
    index = ensure_dataset_index(selection.dataset_name, root=proj_root)
    samples = _select_samples(index.samples, selection)
    if not samples:
        raise ValueError("No samples matched the requested selection.")

    run_dir = _create_run_dir(selection, root=proj_root)
    segmenter = create_segmenter(selection.backend_name)
    manifest: list[dict[str, Any]] = []

    for sample in samples:
        try:
            raw = _load_sample_image(sample)
        except Exception as exc:  # noqa: BLE001
            manifest.append(
                {
                    "sample_id": sample.sample_id,
                    "status": "failed",
                    "reason": str(exc),
                }
            )
            continue

        result = segmenter.segment(raw, sample)
        output = _save_sample_outputs(run_dir, sample, raw, result)
        manifest.append(output)

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "sample_count": len(samples),
        "backend_name": selection.backend_name,
        "dataset_name": selection.dataset_name,
        "selection": asdict(selection),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _select_samples(samples: list[DatasetSample], selection: SegmentationRunSelection) -> list[DatasetSample]:
    selected = samples
    selector_count = sum(
        value is not None
        for value in (
            selection.sample_id,
            selection.split,
            selection.sequence_id,
        )
    )
    if selector_count != 1:
        raise ValueError("Exactly one of sample_id, split, or sequence_id must be provided.")

    if selection.sample_id is not None:
        selected = [sample for sample in samples if sample.sample_id == selection.sample_id]
    elif selection.split is not None:
        selected = [sample for sample in samples if sample.split == selection.split]
    elif selection.sequence_id is not None:
        selected = [sample for sample in samples if sample.sequence_id == selection.sequence_id]

    selected = sorted(selected, key=lambda sample: sample.sample_id.lower())
    if selection.limit > 0:
        selected = selected[: selection.limit]
    return selected


def _create_run_dir(selection: SegmentationRunSelection, root: Path) -> Path:
    base = runs_root(root) / "segment" / selection.backend_name / selection.dataset_name
    if selection.sample_id is not None:
        selector = f"sample_{_safe_name(selection.sample_id)}"
    elif selection.split is not None:
        selector = f"split_{_safe_name(selection.split)}"
    else:
        selector = f"sequence_{_safe_name(selection.sequence_id or 'unknown')}"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(base / f"{selector}_{stamp}")


def _load_sample_image(sample: DatasetSample) -> np.ndarray:
    if sample.image_path:
        return read_image(sample.image_path)
    if sample.video_path:
        frame_index = sample.frame_index or 0
        return read_video_frame(sample.video_path, frame_index)
    raise FileNotFoundError(f"Sample has no image or video source: {sample.sample_id}")


def _save_sample_outputs(
    run_dir: Path,
    sample: DatasetSample,
    raw: np.ndarray,
    result: SegmentationResult,
) -> dict[str, Any]:
    sample_dir = ensure_dir(run_dir / _safe_name(sample.sample_id))
    mask_path = sample_dir / "mask.png"
    overlay_path = sample_dir / "overlay.png"
    result_path = sample_dir / "result.json"

    overlay = _render_overlay(raw, result.mask)
    write_image(mask_path, result.mask)
    write_image(overlay_path, overlay)

    payload = {
        "sample": sample.to_dict(),
        "result": result.to_dict(),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "sample_id": sample.sample_id,
        "status": "ok",
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "result_path": str(result_path),
    }


def _render_overlay(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    raw_bgr = as_bgr(raw)
    mask_color = colorize_mask(mask)
    if mask_color.shape[:2] != raw_bgr.shape[:2]:
        mask_color = cv2.resize(mask_color, (raw_bgr.shape[1], raw_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(raw_bgr, 0.68, mask_color, 0.32, 0.0)
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


def _safe_name(text: str) -> str:
    return text.replace("/", "__").replace("\\", "__").replace(" ", "_").replace(":", "_")
