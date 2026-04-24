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

## Tracking Idea

The initial tracking pipeline will follow this flow:

1. Obtain an instrument segmentation mask for each frame.
2. Analyze connected components, contours, and skeletons.
3. Select the most plausible extremity as the instrument tip.
4. Stabilize the estimate with temporal information from previous frames.
5. Export coordinates and qualitative visualizations.

## Documents

- Detailed implementation plan: [PRD.md](/d:/dev/esis/PRD.md)

## Current Status

The project is currently focused on EndoVis17 and EndoVis18. The next implementation steps should prioritize those two datasets first and then add the four deep-learning segmentation backends behind one shared API.
