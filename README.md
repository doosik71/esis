# ESIS

ESIS is a Python project for fast surgical instrument tip tracking based on surgical instrument segmentation. The initial target is an experiment pipeline that estimates instrument tip locations from segmentation masks and tracks those tips across image sequences or video frames.

## Core Principles

- Do not modify original dataset files.
- Keep most Python source code under `esis/`.
- Store temporary files, caches, logs, and debug outputs under `temp/`.

## Initial Target Datasets

- `data/EndoVis15 Instrument Subchallenge Dataset`
- `data/EndoVis17 Instrument Subchallenge Dataset`
- `data/EndoVis18 Instrument Subchallenge Dataset`
- `data/EndoVis19 Instrument Subchallenge Dataset`

Notes:

- These four datasets are currently present in the local workspace as of 2026-04-24.

## Planned Features

- EndoVis15, EndoVis17, EndoVis18, and EndoVis19 dataset loading
- Instrument segmentation result loading or generation
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
2. Implement dataset adapters for EndoVis15, EndoVis17, EndoVis18, and EndoVis19.
3. Implement a baseline segmentation and tip detection pipeline.
4. Add temporal tracking and visualization.
5. Add benchmarking and experiment utilities.

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

The project is currently in the planning stage. The next step is to create the package skeleton and implement dataset loaders for EndoVis15, EndoVis17, EndoVis18, and EndoVis19.
