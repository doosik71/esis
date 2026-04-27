# ESIS

ESIS is a Python project for fast surgical instrument tip tracking based on surgical instrument segmentation. The initial target is an experiment pipeline that estimates instrument tip locations from segmentation masks and tracks those tips across image sequences or video frames.

## Core Principles

- Do not modify original dataset files.
- Keep most Python source code under `esis/`.
- Store temporary files, caches, logs, and debug outputs under `temp/`.

## Initial Target Datasets

- `data/EndoVis17 Instrument Subchallenge Dataset`
- `data/EndoVis18 Instrument Subchallenge Dataset`

Notes:

- For the current development scope, EndoVis17 and EndoVis18 are the active target datasets.
- EndoVis15 and EndoVis19 are present in the repository, but they are excluded from this development phase.

## Planned Features

- EndoVis17 and EndoVis18 dataset loading
- Instrument segmentation result loading or generation
- Four interchangeable deep-learning segmentation backends with one consistent interface
- Tip localization from segmentation masks
- Temporal tip tracking across frames
- Visualization export and coordinate export
- Speed and localization evaluation

## Expected Project Structure

```text
.
+-- data/
+-- esis/
|   +-- datasets/
|   +-- segmentation/
|   +-- tracking/
|   +-- visualization/
|   +-- evaluation/
|   +-- utils/
|   `-- cli/
+-- temp/
+-- main.py
+-- PRD.md
`-- README.md
```

## Development Plan

1. Prepare project documents and baseline structure.
2. Implement dataset adapters for EndoVis17 and EndoVis18.
3. Implement a unified segmentation interface and four deep-learning segmentation backends.
4. Add temporal tracking and visualization.
5. Add benchmarking and experiment utilities.

## Segmentation Backends

The project will implement the following segmentation backends behind one shared interface.

- `mask_loader`: dataset ground-truth mask loader for validation and debugging
- `adapter_vit_cnn`: ViT + CNN adapter style supervised model
- `matis`: masked-attention transformer style model for surgical instrument segmentation
- `surgsam2`: surgical SAM2-style video segmentation model
- `sam2_zero_shot`: promptable SAM2 zero-shot segmentation mode

The first item is a non-learning baseline for verification. The latter four are the active deep-learning model targets for this project.

## Checkpoints And Weights

Backend weight discovery follows two layers:

- Generic pretrained backbones are downloaded and cached by the underlying framework when needed.
- Task-specific checkpoints are searched under `temp/cache/checkpoints/`.

Current backend behavior:

- `mask_loader`
  - No model weights are required.
- `adapter_vit_cnn`
  - Uses a pretrained DINOv2-style backbone through `timm`.
  - Looks for task-specific checkpoints under `temp/cache/checkpoints/adapter_vit_cnn/<dataset>/`.
- `matis`
  - Uses a pretrained SAM2 Hiera-style backbone through `timm`.
  - Looks for task-specific checkpoints under `temp/cache/checkpoints/matis/<dataset>/`.
  - Fold-specific EndoVis17 checkpoints are expected under `temp/cache/checkpoints/matis/endovis17/fold{n}/`.
- `sam2_zero_shot`
  - Uses `facebook/sam2.1-hiera-tiny` through `transformers`.
  - No local task-specific checkpoint is required for zero-shot inference.
- `surgsam2`
  - Uses the vendored official SurgSAM-2 code under `temp/cache/vendors/Surgical-SAM-2`.
  - Loads official SAM2 weights from Hugging Face when needed.

The detailed local checkpoint convention is documented in:

- [temp/cache/checkpoints/README.md](./temp/cache/checkpoints/README.md)

## Segmentation CLI

The repository now includes a segmentation runner CLI that saves outputs under `temp/runs/`.

Examples:

```bash
uv run python main.py segment run --dataset endovis17 --backend sam2_zero_shot --sample-id train/seq_1_frame000.png
uv run python main.py segment run --dataset endovis17 --backend adapter_vit_cnn --split train --limit 8
uv run python main.py segment run --dataset endovis18 --backend matis --sequence-id sequence_1
```

Output layout:

- `temp/runs/segment/<backend>/<dataset>/<selection>_<timestamp>/`
- one subdirectory per processed sample
- `mask.png`
- `overlay.png`
- `result.json`
- run-level `manifest.json`
- run-level `summary.json`

## Tracking Idea

The initial tracking pipeline will follow this flow:

1. Obtain an instrument segmentation mask for each frame.
2. Analyze connected components, contours, and skeletons.
3. Select the most plausible extremity as the instrument tip.
4. Stabilize the estimate with temporal information from previous frames.
5. Export coordinates and qualitative visualizations.

## Documents

- Detailed implementation plan: [PRD.md](./PRD.md)

## Current Status

The project is currently focused on EndoVis17 and EndoVis18. The next implementation steps should prioritize those two datasets first and then add the four deep-learning segmentation backends behind one shared API.
