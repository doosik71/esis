from __future__ import annotations

import csv
from pathlib import Path

from esis.datasets.base import BaseDatasetAdapter
from esis.datasets.schema import DatasetAdapterError, DatasetSample


class EndoVis19Adapter(BaseDatasetAdapter):
    dataset_name = "endovis19"
    index_file_name = "endovis19_index.json"

    def collect_samples(self) -> list[DatasetSample]:
        data_root = self.dataset_root / "data"
        if not data_root.exists():
            raise DatasetAdapterError(f"Missing dataset data directory: {data_root}")

        samples: list[DatasetSample] = []
        samples.extend(self._collect_raw_video_samples(data_root / "Raw data"))
        samples.extend(self._collect_release_clip_samples(data_root / "ROBUST-MIS-2019-RELEASE-06082019"))
        return samples

    def _collect_raw_video_samples(self, raw_root: Path) -> list[DatasetSample]:
        if not raw_root.exists():
            return []

        samples: list[DatasetSample] = []
        for procedure_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
            for case_dir in sorted(path for path in procedure_dir.iterdir() if path.is_dir()):
                video_path = next((p for p in sorted(case_dir.glob("*.avi")) if p.is_file()), None)
                device_csv = next((p for p in sorted(case_dir.glob("*device*.csv")) if p.is_file()), None)
                if video_path is None:
                    continue
                case_name = f"{procedure_dir.name}/{case_dir.name}"
                samples.append(
                    DatasetSample(
                        dataset_name=self.dataset_name,
                        split="raw",
                        sample_id=f"raw/{case_name}",
                        sequence_id=f"raw/{case_name}",
                        modality="video_sequence",
                        video_path=str(video_path),
                        annotations={"device_csv": str(device_csv)} if device_csv else {},
                        metadata={
                            "source": "raw_data",
                            "procedure": procedure_dir.name,
                            "case_id": case_dir.name,
                        },
                    )
                )
        return samples

    def _collect_release_clip_samples(self, release_root: Path) -> list[DatasetSample]:
        if not release_root.exists():
            return []

        samples: list[DatasetSample] = []
        for manifest_path in sorted(release_root.rglob("synapse_metadata_manifest.tsv")):
            if not manifest_path.is_file():
                continue
            if "venv" in manifest_path.parts:
                continue
            rows = self.parse_tsv_rows(manifest_path)
            parent_parts = manifest_path.parent.relative_to(release_root).parts
            if not parent_parts:
                continue

            clip_files: dict[str, dict[str, str]] = {}
            for row in rows:
                path_text = row.get("path", "").strip()
                file_name = row.get("name", "").strip()
                clip_id = self._clip_id_from_path(path_text)
                if not clip_id or not file_name:
                    continue
                clip_entry = clip_files.setdefault(clip_id, {})
                clip_entry[file_name] = path_text

            for clip_id, file_map in clip_files.items():
                split = parent_parts[0].lower()
                procedure = parent_parts[1] if len(parent_parts) > 1 else "unknown"
                case_id = parent_parts[2] if len(parent_parts) > 2 else "unknown"
                stage = parent_parts[1] if split == "testing" and len(parent_parts) > 1 else None
                procedure_name = parent_parts[2] if split == "testing" and len(parent_parts) > 2 else procedure
                case_name = parent_parts[3] if split == "testing" and len(parent_parts) > 3 else case_id
                sequence_id = f"{split}/{procedure_name}/{case_name}"
                sample_id = f"{sequence_id}/{clip_id}"
                timestamp_ms = int(clip_id) if clip_id.isdigit() else None

                samples.append(
                    DatasetSample(
                        dataset_name=self.dataset_name,
                        split=split,
                        sample_id=sample_id,
                        sequence_id=sequence_id,
                        modality="clip_manifest",
                        image_path=file_map.get("raw.png"),
                        label_path=file_map.get("instrument_instances.png"),
                        video_path=file_map.get("10s_video.zip"),
                        timestamp_ms=timestamp_ms,
                        metadata={
                            "source": "release_manifest",
                            "manifest_path": str(manifest_path),
                            "procedure": procedure_name,
                            "case_id": case_name,
                            "stage": stage,
                            "available_files": sorted(file_map),
                        },
                    )
                )
        return samples

    def _clip_id_from_path(self, path_text: str) -> str | None:
        normalized = path_text.replace("\\", "/").rstrip("/")
        if not normalized:
            return None
        return normalized.split("/")[-1] if "/" in normalized else None
