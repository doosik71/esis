from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from esis.datasets.schema import DatasetSample
from esis.segmentation.base import BaseSegmenter, SegmentationResult
from esis.segmentation.checkpoints import CheckpointResolution, resolve_matis_checkpoint
from esis.segmentation.torch_utils import resolve_device, vendor_path
from esis.utils.config import cache_root, project_root


@dataclass(slots=True)
class MatisConfig:
    checkpoint_path: str | None = None
    dataset_name: str | None = None
    fold: int | None = None
    device: str | None = None
    test_crop_size: int = 224
    max_boxes: int = 10
    config_path: str | None = None
    features_train_path: str | None = None
    features_val_path: str | None = None


@dataclass(slots=True)
class _MatisAssets:
    config_path: str
    checkpoint: CheckpointResolution
    feature_paths: list[str]
    searched_feature_paths: list[str] = field(default_factory=list)


class MatisSegmenter(BaseSegmenter):
    name = "matis"

    def __init__(self, config: MatisConfig | None = None) -> None:
        self.config = config or MatisConfig()
        self.device = resolve_device(self.config.device)
        self.vendor_root = cache_root() / "vendors" / "MATIS"
        self.checkpoint_resolution = resolve_matis_checkpoint(
            dataset_name=self.config.dataset_name,
            explicit_path=self.config.checkpoint_path,
            fold=self._effective_fold(self.config.dataset_name, self.config.fold),
        )

        self.model: Any | None = None
        self.runtime_cfg: Any | None = None
        self.runtime_assets: _MatisAssets | None = None
        self.features_by_name: dict[str, dict[str, Any]] | None = None
        self.runtime_dataset_name: str | None = None
        self.checkpoint_loaded = False
        self.load_error: str | None = None

    @torch.inference_mode()
    def segment(self, image: np.ndarray, sample: DatasetSample | None = None) -> SegmentationResult:
        dataset_name = self._resolve_dataset_name(sample)
        self._ensure_runtime(dataset_name)

        assert self.runtime_cfg is not None
        assert self.features_by_name is not None
        assert self.model is not None

        feature_entry = self._find_feature_entry(sample)
        if feature_entry is None:
            raise RuntimeError(
                "MATIS feature entry was not found for the selected sample. "
                "The official MATIS backend needs the precomputed feature bundle "
                "from the upstream MATIS release, keyed by frame filename."
            )

        image_paths = self._build_clip_image_paths(sample)
        boxes, box_keys, box_features = self._extract_feature_proposals(feature_entry)
        inputs, boxes_tensor, boxes_mask_tensor, feature_tensor = self._prepare_inputs(
            image_paths=image_paths,
            boxes=boxes,
            box_features=box_features,
        )

        predictions = self.model(inputs, boxes_tensor, feature_tensor, boxes_mask_tensor)
        tool_probabilities = predictions["tools"][0].detach().cpu().numpy()
        semantic_mask = self._decode_semantic_mask(
            probabilities=tool_probabilities,
            box_keys=box_keys,
            feature_entry=feature_entry,
            output_shape=image.shape[:2],
        )

        sample_id = sample.sample_id if sample is not None else "unknown"
        return SegmentationResult(
            sample_id=sample_id,
            mask=semantic_mask.astype(np.uint8),
            metadata={
                "segmenter": self.name,
                "model_type": "official_matis",
                "model_name": "MViT",
                "checkpoint_loaded": self.checkpoint_loaded,
                "checkpoint_path": self.checkpoint_resolution.checkpoint_path,
                "checkpoint_source": self.checkpoint_resolution.source,
                "config_path": self.runtime_assets.config_path if self.runtime_assets is not None else None,
                "feature_paths": self.runtime_assets.feature_paths if self.runtime_assets is not None else [],
                "device": str(self.device),
                "proposal_count": int(len(box_keys)),
                "dataset_name": dataset_name,
                "load_error": self.load_error,
            },
        )

    def _ensure_runtime(self, dataset_name: str) -> None:
        if self.runtime_dataset_name == dataset_name and self.model is not None and self.features_by_name is not None:
            return

        try:
            self.runtime_assets = self._resolve_assets(dataset_name)
            self.runtime_cfg = self._build_runtime_cfg(dataset_name, self.runtime_assets)
            self.model = self._build_model(self.runtime_cfg, self.runtime_assets)
            self.features_by_name = self._load_feature_index(self.runtime_assets.feature_paths)
            self.runtime_dataset_name = dataset_name
            self.checkpoint_loaded = True
            self.load_error = None
        except Exception as exc:
            self.checkpoint_loaded = False
            self.runtime_dataset_name = dataset_name
            self.load_error = str(exc)
            raise

    def _build_model(self, cfg: Any, assets: _MatisAssets) -> Any:
        self._validate_vendor_root()
        with vendor_path(self.vendor_root):
            try:
                from matis.models import build_model
                from matis.utils.checkpoint import load_checkpoint
            except ModuleNotFoundError as exc:
                dependency = getattr(exc, "name", "unknown dependency")
                raise RuntimeError(
                    "Official MATIS dependencies are missing. "
                    f"Install the MATIS runtime requirements first. Missing module: {dependency}"
                ) from exc

            model = build_model(cfg)
            model = model.to(self.device)
            load_checkpoint(
                assets.checkpoint.checkpoint_path,
                model,
                data_parallel=False,
            )
            model.eval()
            return model

    def _build_runtime_cfg(self, dataset_name: str, assets: _MatisAssets) -> Any:
        self._validate_vendor_root()
        with vendor_path(self.vendor_root):
            try:
                from matis.config.defaults import assert_and_infer_cfg, get_cfg
            except ModuleNotFoundError as exc:
                dependency = getattr(exc, "name", "unknown dependency")
                raise RuntimeError(
                    "Official MATIS config dependencies are missing. "
                    f"Install the MATIS runtime requirements first. Missing module: {dependency}"
                ) from exc

            cfg = get_cfg()
            cfg.merge_from_file(assets.config_path)

            cfg.NUM_GPUS = 0
            cfg.DATA.JUST_CENTER = True
            cfg.DATA.TEST_CROP_SIZE = self.config.test_crop_size
            cfg.DATA.TRAIN_CROP_SIZE = self.config.test_crop_size
            cfg.DATA.MAX_BBOXES = self.config.max_boxes
            cfg.DETECTION.ENABLE = True
            cfg.TRAIN.ENABLE = False
            cfg.TEST.ENABLE = False
            cfg.TRAIN.BATCH_SIZE = 1
            cfg.TEST.BATCH_SIZE = 1
            cfg.TRAIN.DATASET = self._official_dataset_name(dataset_name)
            cfg.TEST.DATASET = self._official_dataset_name(dataset_name)
            cfg.TASKS.TASKS = ["tools"]
            cfg.TASKS.NUM_CLASSES = [7]
            cfg.TASKS.LOSS_FUNC = ["cross_entropy"]
            cfg.TASKS.HEAD_ACT = ["softmax"]
            cfg.MASKFORMER.ENABLE = True
            cfg.MASKFORMER.FEATURES_TRAIN = assets.feature_paths[0]
            cfg.MASKFORMER.FEATURES_VAL = assets.feature_paths[-1]
            cfg.OUTPUT_DIR = str(project_root() / "temp" / "runs" / "matis_runtime")

            return assert_and_infer_cfg(cfg)

    def _load_feature_index(self, feature_paths: list[str]) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for feature_path in feature_paths:
            payload = torch.load(feature_path, map_location="cpu")
            features = payload.get("features")
            if not isinstance(features, list):
                raise RuntimeError(f"MATIS feature file has an unsupported format: {feature_path}")

            for entry in features:
                file_name = Path(str(entry["file_name"])).name
                indexed[file_name] = entry
                indexed[Path(file_name).stem] = entry
        return indexed

    def _resolve_assets(self, dataset_name: str) -> _MatisAssets:
        config_path = self._resolve_config_path(dataset_name)
        checkpoint = resolve_matis_checkpoint(
            dataset_name=dataset_name,
            explicit_path=self.config.checkpoint_path,
            fold=self._effective_fold(dataset_name, self.config.fold),
        )
        if checkpoint.checkpoint_path is None:
            searched = "\n".join(checkpoint.searched_paths)
            raise RuntimeError(
                "Official MATIS checkpoint was not found.\n"
                "Searched paths:\n"
                f"{searched}"
            )

        feature_paths, searched_feature_paths = self._resolve_feature_paths(dataset_name)
        if not feature_paths:
            searched = "\n".join(searched_feature_paths)
            raise RuntimeError(
                "Official MATIS feature bundle was not found. "
                "The upstream MATIS model requires precomputed proposal features and segments.\n"
                "Searched paths:\n"
                f"{searched}"
            )

        self.checkpoint_resolution = checkpoint
        return _MatisAssets(
            config_path=str(config_path),
            checkpoint=checkpoint,
            feature_paths=feature_paths,
            searched_feature_paths=searched_feature_paths,
        )

    def _resolve_feature_paths(self, dataset_name: str) -> tuple[list[str], list[str]]:
        dataset_token = self._official_dataset_name(dataset_name)
        checkpoint_base = cache_root() / "checkpoints" / "matis"
        candidates: list[Path] = []

        explicit_paths = [self.config.features_train_path, self.config.features_val_path]
        for explicit in explicit_paths:
            if explicit:
                candidates.append(Path(explicit))

        if dataset_name == "endovis17":
            candidates.extend(
                [
                    project_root() / "temp" / "model" / "features_train.pth",
                    project_root() / "temp" / "model" / "features_val.pth",
                    checkpoint_base / "official_bundle" / dataset_token / "features" / "features_train.pth",
                    checkpoint_base / "official_bundle" / dataset_token / "features" / "features_val.pth",
                    checkpoint_base / dataset_name / "features" / "features_train.pth",
                    checkpoint_base / dataset_name / "features" / "features_val.pth",
                ]
            )
        elif dataset_name == "endovis18":
            candidates.extend(
                [
                    project_root() / "temp" / "model" / "features_decoder_train.pth",
                    project_root() / "temp" / "model" / "features_decoder_val.pth",
                    checkpoint_base / "official_bundle" / dataset_token / "features" / "features_decoder_train.pth",
                    checkpoint_base / "official_bundle" / dataset_token / "features" / "features_decoder_val.pth",
                    checkpoint_base / dataset_name / "features" / "features_decoder_train.pth",
                    checkpoint_base / dataset_name / "features" / "features_decoder_val.pth",
                ]
            )
        else:
            raise ValueError(f"Unsupported MATIS dataset: {dataset_name}")

        resolved = [str(path) for path in candidates if path.exists() and path.is_file()]
        return resolved, [str(path) for path in candidates]

    def _resolve_config_path(self, dataset_name: str) -> Path:
        if self.config.config_path:
            path = Path(self.config.config_path)
            if not path.exists():
                raise RuntimeError(f"MATIS config file does not exist: {path}")
            return path

        config_name = "MATIS_FULL.yaml"
        official_dataset = self._official_dataset_name(dataset_name)
        path = self.vendor_root / "configs" / official_dataset / config_name
        if not path.exists():
            raise RuntimeError(f"Official MATIS config file was not found: {path}")
        return path

    def _find_feature_entry(self, sample: DatasetSample | None) -> dict[str, Any] | None:
        if sample is None or sample.image_path is None or self.features_by_name is None:
            return None

        filename = Path(sample.image_path).name
        stem = Path(filename).stem
        candidates = [
            filename,
            stem,
            f"{stem}.png",
            f"{stem}.bmp",
            f"{stem}.jpg",
        ]
        for candidate in candidates:
            entry = self.features_by_name.get(candidate)
            if entry is not None:
                return entry
        return None

    def _extract_feature_proposals(
        self,
        feature_entry: dict[str, Any],
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        proposal_map = feature_entry.get("bboxes")
        if not isinstance(proposal_map, dict) or not proposal_map:
            raise RuntimeError("MATIS feature entry does not contain proposal features.")

        box_keys = list(proposal_map.keys())[: self.config.max_boxes]
        boxes = np.array([self._parse_box_key(key) for key in box_keys], dtype=np.float32)
        box_features = np.array([proposal_map[key] for key in box_keys], dtype=np.float32)
        return boxes, box_keys, box_features

    def _prepare_inputs(
        self,
        image_paths: list[str],
        boxes: np.ndarray,
        box_features: np.ndarray,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.runtime_cfg is not None
        self._validate_vendor_root()

        with vendor_path(self.vendor_root):
            from matis.datasets import cv2_transform
            from matis.datasets import utils as dataset_utils

            images = [self._load_image(path) for path in image_paths]
            if any(image is None for image in images):
                missing = [path for path, image in zip(image_paths, images) if image is None]
                raise RuntimeError(f"MATIS could not read one or more clip frames: {missing}")

            height, width, _ = images[0].shape
            processed_boxes = boxes.copy()
            processed_boxes[:, [0, 2]] *= width
            processed_boxes[:, [1, 3]] *= height
            processed_boxes = cv2_transform.clip_boxes_to_image(processed_boxes, height, width)
            box_list = [processed_boxes]

            scaled_images = [cv2_transform.scale(self.runtime_cfg.DATA.TEST_CROP_SIZE, image) for image in images]
            box_list = [
                cv2_transform.scale_boxes(
                    self.runtime_cfg.DATA.TEST_CROP_SIZE,
                    box_list[0],
                    height,
                    width,
                )
            ]
            scaled_images, box_list = cv2_transform.spatial_shift_crop_list(
                self.runtime_cfg.DATA.TEST_CROP_SIZE,
                scaled_images,
                1,
                boxes=box_list,
            )

            chw_images = [cv2_transform.HWC2CHW(image) / 255.0 for image in scaled_images]
            chw_images = [
                cv2_transform.color_normalization(
                    np.ascontiguousarray(image).astype(np.float32),
                    np.array(self.runtime_cfg.DATA.MEAN, dtype=np.float32),
                    np.array(self.runtime_cfg.DATA.STD, dtype=np.float32),
                )
                for image in chw_images
            ]
            stacked = np.concatenate([np.expand_dims(image, axis=1) for image in chw_images], axis=1)
            if not self.runtime_cfg.AVA.BGR:
                stacked = stacked[::-1, ...]
            stacked = np.ascontiguousarray(stacked)
            frames_tensor = torch.from_numpy(stacked)
            processed_boxes = cv2_transform.clip_boxes_to_image(
                box_list[0],
                frames_tensor[0].shape[1],
                frames_tensor[0].shape[2],
            )

            frame_list = dataset_utils.pack_pathway_output(self.runtime_cfg, frames_tensor)
            frame_list = [frames.unsqueeze(0).to(self.device, dtype=torch.float32) for frames in frame_list]

        proposal_count = len(processed_boxes)
        boxes_padded = np.zeros((self.config.max_boxes, 4), dtype=np.float32)
        features_padded = np.zeros((self.config.max_boxes, 256), dtype=np.float32)
        boxes_mask = np.zeros((self.config.max_boxes,), dtype=bool)

        boxes_padded[:proposal_count] = processed_boxes[:proposal_count]
        features_padded[:proposal_count] = box_features[:proposal_count]
        boxes_mask[:proposal_count] = True

        boxes_tensor = torch.from_numpy(boxes_padded).unsqueeze(0).to(self.device, dtype=torch.float32)
        boxes_mask_tensor = torch.from_numpy(boxes_mask).unsqueeze(0).to(self.device)
        feature_tensor = torch.from_numpy(features_padded).unsqueeze(0).to(self.device, dtype=torch.float32)
        return frame_list, boxes_tensor, boxes_mask_tensor, feature_tensor

    def _decode_semantic_mask(
        self,
        probabilities: np.ndarray,
        box_keys: list[str],
        feature_entry: dict[str, Any],
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        try:
            import pycocotools.mask as coco_mask
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Official MATIS mask decoding requires `pycocotools`. "
                "Install the MATIS runtime dependencies first."
            ) from exc

        segments = feature_entry.get("segments")
        if not isinstance(segments, dict):
            raise RuntimeError("MATIS feature entry does not contain proposal segments.")

        semantic_mask = np.zeros((int(feature_entry["height"]), int(feature_entry["width"])), dtype=np.uint8)
        ranked_instances: list[tuple[float, int, Any]] = []
        for probability, box_key in zip(probabilities, box_keys):
            category = int(np.argmax(probability)) + 1
            score = float(np.max(probability))
            if category < 1 or category > 7:
                continue
            segment = segments.get(box_key)
            if segment is None:
                continue
            ranked_instances.append((score, category, segment))

        ranked_instances.sort(key=lambda item: item[0])
        for _, category, segment in ranked_instances:
            decoded = coco_mask.decode(segment)
            semantic_mask[np.asarray(decoded) == 1] = category

        if semantic_mask.shape != output_shape:
            semantic_mask = cv2.resize(
                semantic_mask,
                (output_shape[1], output_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return semantic_mask

    def _build_clip_image_paths(self, sample: DatasetSample | None) -> list[str]:
        if sample is None or sample.image_path is None:
            raise RuntimeError("MATIS requires a frame-backed sample with a valid image path.")

        assert self.runtime_cfg is not None
        image_path = Path(sample.image_path)
        if not image_path.exists():
            raise RuntimeError(f"Sample image does not exist: {image_path}")

        sibling_frames = sorted(
            image_path.parent.glob(f"{self._sequence_prefix(image_path.name)}*{image_path.suffix}"),
            key=lambda path: path.name,
        )
        if not sibling_frames:
            sibling_frames = sorted(image_path.parent.glob(f"*{image_path.suffix}"), key=lambda path: path.name)

        if not sibling_frames:
            raise RuntimeError(f"MATIS could not discover sibling frames for: {image_path}")

        center_index = self._resolve_center_index(sample, sibling_frames, image_path)

        self._validate_vendor_root()
        with vendor_path(self.vendor_root):
            from matis.datasets import utils as dataset_utils

            seq = dataset_utils.get_sequence(
                center_index,
                (self.runtime_cfg.DATA.NUM_FRAMES * self.runtime_cfg.DATA.SAMPLING_RATE) // 2,
                self.runtime_cfg.DATA.SAMPLING_RATE,
                num_frames=len(sibling_frames),
            )
        return [str(sibling_frames[index]) for index in seq]

    def _resolve_center_index(self, sample: DatasetSample, siblings: list[Path], image_path: Path) -> int:
        if sample.frame_index is not None and 0 <= sample.frame_index < len(siblings):
            return sample.frame_index
        try:
            return siblings.index(image_path)
        except ValueError:
            for index, sibling in enumerate(siblings):
                if sibling.name == image_path.name:
                    return index
        raise RuntimeError(f"MATIS could not locate the selected frame in its sibling sequence: {image_path}")

    def _resolve_dataset_name(self, sample: DatasetSample | None) -> str:
        dataset_name = sample.dataset_name if sample is not None else self.config.dataset_name
        if dataset_name not in {"endovis17", "endovis18"}:
            raise ValueError("Official MATIS support is currently limited to endovis17 and endovis18.")
        return dataset_name

    def _official_dataset_name(self, dataset_name: str) -> str:
        if dataset_name == "endovis17":
            return "endovis_2017"
        if dataset_name == "endovis18":
            return "endovis_2018"
        raise ValueError(f"Unsupported MATIS dataset: {dataset_name}")

    def _effective_fold(self, dataset_name: str | None, requested_fold: int | None) -> int | None:
        if requested_fold is not None:
            return requested_fold
        if dataset_name == "endovis17":
            return 3
        return None

    def _sequence_prefix(self, filename: str) -> str:
        if "_frame" in filename:
            return filename.split("_frame", 1)[0] + "_frame"
        return Path(filename).stem

    def _load_image(self, path: str) -> np.ndarray | None:
        return cv2.imread(path, cv2.IMREAD_COLOR)

    def _parse_box_key(self, box_key: str) -> list[float]:
        return [float(value) for value in box_key.split()]

    def _validate_vendor_root(self) -> None:
        if not self.vendor_root.exists():
            raise RuntimeError(
                "Official MATIS vendor repository was not found under "
                f"{self.vendor_root}. Clone or restore the upstream MATIS repo first."
            )

