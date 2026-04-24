from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from esis.datasets.registry import build_dataset_index
from esis.datasets.schema import DatasetIndex, DatasetSample
from esis.utils.config import debug_root, default_dataset_roots, index_root, project_root
from esis.utils.io import annotate_image, as_bgr, colorize_mask, ensure_dir, read_first_video_frame, read_image, write_image


def load_dataset_index(dataset_name: str, root: Path | None = None) -> DatasetIndex:
    proj_root = root or project_root()
    index_path = index_root(proj_root) / f"{dataset_name}_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    samples = [DatasetSample(**sample) for sample in payload["samples"]]
    payload["samples"] = samples
    return DatasetIndex(**payload)


def ensure_dataset_index(dataset_name: str, rebuild: bool = False, root: Path | None = None) -> DatasetIndex:
    proj_root = root or project_root()
    path = index_root(proj_root) / f"{dataset_name}_index.json"
    if path.exists() and not rebuild:
        return load_dataset_index(dataset_name, proj_root)
    dataset_root = default_dataset_roots(proj_root)[dataset_name]
    return build_dataset_index(dataset_name, dataset_root, project_root=proj_root)


def summarize_index(index: DatasetIndex) -> dict[str, Any]:
    samples_with_image = sum(1 for sample in index.samples if sample.image_path)
    samples_with_label = sum(1 for sample in index.samples if sample.label_path)
    samples_with_video = sum(1 for sample in index.samples if sample.video_path)
    previewable_samples = sum(1 for sample in index.samples if _resolve_preview_path(sample))
    annotation_keys = sorted({key for sample in index.samples for key in sample.annotations})
    return {
        "dataset_name": index.dataset_name,
        "dataset_root": index.dataset_root,
        "sample_count": index.sample_count,
        "sequence_count": index.sequence_count,
        "samples_with_image": samples_with_image,
        "samples_with_label": samples_with_label,
        "samples_with_video": samples_with_video,
        "previewable_samples": previewable_samples,
        "split_counts": index.summary.get("split_counts", {}),
        "modality_counts": index.summary.get("modality_counts", {}),
        "annotation_keys": annotation_keys,
        "metadata_files": len(index.metadata_files),
    }


def validate_index(index: DatasetIndex, max_samples: int = 64) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    inspected: list[dict[str, Any]] = []
    samples = index.samples[:max_samples] if max_samples > 0 else index.samples
    for sample in samples:
        sample_report = _inspect_sample(sample)
        inspected.append(sample_report)
        issues.extend(sample_report["issues"])
    return {
        "dataset_name": index.dataset_name,
        "validated_samples": len(samples),
        "issue_count": len(issues),
        "issues": issues,
        "sample_reports": inspected,
    }


def export_dataset_debug(
    dataset_name: str,
    rebuild_index: bool = False,
    preview_count: int = 6,
    validate_count: int = 64,
    root: Path | None = None,
) -> dict[str, Path]:
    proj_root = root or project_root()
    index = ensure_dataset_index(dataset_name, rebuild=rebuild_index, root=proj_root)
    debug_dir = ensure_dir(debug_root(proj_root) / dataset_name)

    summary = summarize_index(index)
    validation = validate_index(index, max_samples=validate_count)
    previews = export_previews(index, debug_dir, limit=preview_count)

    summary_path = debug_dir / "summary.json"
    report_path = debug_dir / "report.md"
    validation_path = debug_dir / "validation.json"
    preview_manifest_path = debug_dir / "preview_manifest.json"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    preview_manifest_path.write_text(json.dumps(previews, indent=2), encoding="utf-8")
    report_path.write_text(render_markdown_report(summary, validation, previews), encoding="utf-8")

    return {
        "summary": summary_path,
        "report": report_path,
        "validation": validation_path,
        "preview_manifest": preview_manifest_path,
        "preview_dir": debug_dir / "previews",
    }


def export_previews(index: DatasetIndex, output_dir: Path, limit: int = 6) -> list[dict[str, Any]]:
    preview_dir = ensure_dir(output_dir / "previews")
    exported: list[dict[str, Any]] = []
    for sample in index.samples:
        if len(exported) >= limit:
            break
        preview_path = _resolve_preview_path(sample)
        if preview_path is None:
            continue
        try:
            image = _load_preview_image(sample, preview_path)
        except Exception as exc:  # noqa: BLE001
            exported.append(
                {
                    "sample_id": sample.sample_id,
                    "status": "failed",
                    "reason": str(exc),
                }
            )
            continue

        preview_image = _compose_preview(sample, image)
        target = preview_dir / f"{_safe_name(sample.sample_id)}.png"
        write_image(target, preview_image)
        exported.append(
            {
                "sample_id": sample.sample_id,
                "status": "ok",
                "output_path": str(target),
            }
        )
    return exported


def render_markdown_report(summary: dict[str, Any], validation: dict[str, Any], previews: list[dict[str, Any]]) -> str:
    lines = [
        f"# Dataset Report: {summary['dataset_name']}",
        "",
        "## Summary",
        "",
        f"- Samples: {summary['sample_count']}",
        f"- Sequences: {summary['sequence_count']}",
        f"- Samples with images: {summary['samples_with_image']}",
        f"- Samples with labels: {summary['samples_with_label']}",
        f"- Samples with videos: {summary['samples_with_video']}",
        f"- Previewable samples: {summary['previewable_samples']}",
        f"- Metadata files: {summary['metadata_files']}",
        "",
        "## Splits",
        "",
    ]
    for split, count in sorted(summary["split_counts"].items()):
        lines.append(f"- {split}: {count}")
    lines.extend(["", "## Modalities", ""])
    for modality, count in sorted(summary["modality_counts"].items()):
        lines.append(f"- {modality}: {count}")
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Validated samples: {validation['validated_samples']}",
            f"- Issue count: {validation['issue_count']}",
            "",
            "## Preview Export",
            "",
        ]
    )
    for item in previews:
        label = item["sample_id"]
        status = item["status"]
        lines.append(f"- {label}: {status}")
    return "\n".join(lines) + "\n"


def _inspect_sample(sample: DatasetSample) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    media = {}

    if sample.image_path:
        media["image"] = _probe_path(sample.image_path, required=sample.metadata.get("source") != "release_manifest")
        if media["image"].get("exists"):
            shape, dtype = _array_signature(read_image(sample.image_path))
            media["image"]["shape"] = shape
            media["image"]["dtype"] = dtype

    if sample.label_path:
        media["label"] = _probe_path(sample.label_path, required=sample.metadata.get("source") != "release_manifest")
        if media["label"].get("exists"):
            shape, dtype = _array_signature(read_image(sample.label_path))
            media["label"]["shape"] = shape
            media["label"]["dtype"] = dtype

    if sample.video_path:
        required = not str(sample.video_path).lower().endswith(".zip")
        media["video"] = _probe_path(sample.video_path, required=required)
        if media["video"].get("exists") and str(sample.video_path).lower().endswith(".avi"):
            frame = read_first_video_frame(sample.video_path)
            shape, dtype = _array_signature(frame)
            media["video"]["shape"] = shape
            media["video"]["dtype"] = dtype

    for name, path_text in sample.annotations.items():
        probe = _probe_path(path_text, required=False)
        media[f"annotation:{name}"] = probe

    if sample.image_path and sample.label_path:
        image_path = Path(sample.image_path)
        label_path = Path(sample.label_path)
        if image_path.exists() and label_path.exists():
            image = read_image(image_path)
            label = read_image(label_path)
            if image.shape[:2] != label.shape[:2]:
                issues.append(
                    {
                        "sample_id": sample.sample_id,
                        "severity": "error",
                        "message": f"Image/label shape mismatch: {image.shape[:2]} vs {label.shape[:2]}",
                    }
                )

    for key, probe in media.items():
        if probe.get("severity"):
            issues.append(
                {
                    "sample_id": sample.sample_id,
                    "severity": probe["severity"],
                    "message": f"{key} path check failed: {probe['message']}",
                }
            )

    return {
        "sample_id": sample.sample_id,
        "sequence_id": sample.sequence_id,
        "split": sample.split,
        "media": media,
        "issues": issues,
    }


def _probe_path(path_text: str, required: bool) -> dict[str, Any]:
    path = Path(path_text)
    exists = path.exists()
    result = {
        "path": path_text,
        "exists": exists,
    }
    if not exists and required:
        result["severity"] = "error"
        result["message"] = "missing path"
    elif not exists:
        result["severity"] = "warning"
        result["message"] = "path not materialized in local workspace"
    return result


def _array_signature(array: np.ndarray) -> tuple[list[int], str]:
    return list(array.shape), str(array.dtype)


def _resolve_preview_path(sample: DatasetSample) -> str | None:
    if sample.image_path and Path(sample.image_path).exists():
        return sample.image_path
    if sample.video_path and str(sample.video_path).lower().endswith(".avi") and Path(sample.video_path).exists():
        return sample.video_path
    return None


def _load_preview_image(sample: DatasetSample, preview_path: str) -> np.ndarray:
    if sample.image_path == preview_path:
        return read_image(preview_path)
    return read_first_video_frame(preview_path)


def _compose_preview(sample: DatasetSample, image: np.ndarray) -> np.ndarray:
    base = as_bgr(image)
    label_preview = None
    if sample.label_path and Path(sample.label_path).exists():
        label = read_image(sample.label_path)
        label_preview = colorize_mask(label)
        if label_preview.shape[:2] != base.shape[:2]:
            label_preview = cv2.resize(label_preview, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)

    if label_preview is not None:
        combined = np.concatenate([base, label_preview], axis=1)
    else:
        combined = base

    lines = [
        sample.dataset_name,
        sample.sample_id,
        f"split={sample.split} modality={sample.modality}",
    ]
    return annotate_image(combined, lines)


def _safe_name(text: str) -> str:
    return text.replace("/", "__").replace("\\", "__").replace(" ", "_").replace(":", "_")
