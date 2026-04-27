from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from esis.utils.config import checkpoint_root, project_root


@dataclass(slots=True)
class CheckpointResolution:
    backend_name: str
    dataset_name: str | None
    checkpoint_path: str | None
    source: str
    searched_paths: list[str]


def _existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def resolve_adapter_vit_cnn_checkpoint(
    dataset_name: str | None = None,
    explicit_path: str | None = None,
) -> CheckpointResolution:
    searched: list[Path] = []
    if explicit_path:
        explicit = Path(explicit_path)
        searched.append(explicit)
        if explicit.exists():
            return CheckpointResolution(
                backend_name="adapter_vit_cnn",
                dataset_name=dataset_name,
                checkpoint_path=str(explicit),
                source="explicit",
                searched_paths=[str(path) for path in searched],
            )

    root = checkpoint_root()
    backend_root = root / "adapter_vit_cnn"
    dataset_token = (dataset_name or "generic").lower()
    candidates = [
        backend_root / dataset_token / "checkpoint.pth.tar",
        backend_root / dataset_token / "checkpoint.pth",
        backend_root / dataset_token / "model.pth",
        backend_root / dataset_token / f"adapter_vit_cnn_{dataset_token}.pth",
        backend_root / dataset_token / f"adapter_vit_cnn_{dataset_token}.pt",
        backend_root / "converted" / dataset_token / "checkpoint.pth.tar",
        backend_root / "converted" / dataset_token / "checkpoint.pth",
    ]
    searched.extend(candidates)
    resolved = _existing_path(candidates)
    return CheckpointResolution(
        backend_name="adapter_vit_cnn",
        dataset_name=dataset_name,
        checkpoint_path=str(resolved) if resolved is not None else None,
        source="convention" if resolved is not None else "missing",
        searched_paths=[str(path) for path in searched],
    )


def resolve_matis_checkpoint(
    dataset_name: str | None = None,
    explicit_path: str | None = None,
    fold: int | None = None,
) -> CheckpointResolution:
    searched: list[Path] = []
    if explicit_path:
        explicit = Path(explicit_path)
        searched.append(explicit)
        if explicit.exists():
            return CheckpointResolution(
                backend_name="matis",
                dataset_name=dataset_name,
                checkpoint_path=str(explicit),
                source="explicit",
                searched_paths=[str(path) for path in searched],
            )

    workspace_default = project_root() / "temp" / "model" / "matis_pretrained_model.pyth"
    root = checkpoint_root()
    backend_root = root / "matis"
    dataset_token = (dataset_name or "generic").lower()
    fold_name = f"fold{fold}" if fold is not None else "default"
    candidates = [
        workspace_default,
        backend_root / dataset_token / fold_name / "matis_pretrained_model.pyth",
        backend_root / dataset_token / "matis_pretrained_model.pyth",
        backend_root / dataset_token / fold_name / "checkpoint.pyth",
        backend_root / dataset_token / fold_name / "checkpoint.pth",
        backend_root / "official_bundle" / "endovis_2017" / "models" / f"Fold{fold or 3}" / "matis_pretrained_model.pyth",
        backend_root / "official_bundle" / "endovis_2018" / "models" / "matis_pretrained_model.pyth",
    ]
    searched.extend(candidates)
    resolved = _existing_path(candidates)
    return CheckpointResolution(
        backend_name="matis",
        dataset_name=dataset_name,
        checkpoint_path=str(resolved) if resolved is not None else None,
        source="convention" if resolved is not None else "missing",
        searched_paths=[str(path) for path in searched],
    )
