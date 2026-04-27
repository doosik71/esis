# PRD: Surgical Instrument Tip Tracking

## 1. Overview

This project aims to build a Python program that tracks the tip of surgical instruments quickly from images or video frames by using surgical instrument segmentation.

The target datasets for this project are:

- `data/EndoVis17 Instrument Subchallenge Dataset`
- `data/EndoVis18 Instrument Subchallenge Dataset`

Datasets excluded from the current development scope:

- `data/EndoVis15 Instrument Subchallenge Dataset`
- `data/EndoVis19 Instrument Subchallenge Dataset`

Important constraints:

- Files inside dataset folders are shared public data and must not be modified.
- Most Python modules should be implemented under `esis/`.
- Temporary files, caches, logs, debug outputs, and intermediate artifacts should be stored under `temp/`.

## 2. Problem Statement

Tracking the instrument tip in surgical video is a core capability for robotic surgery support, quantitative video analysis, skill assessment, and safety monitoring. This project will use segmentation results to identify instrument regions, estimate tip positions from instrument geometry, and stabilize those estimates across frames for fast and robust tracking.

## 3. Goals

### 3.1 Primary goals

- Build common dataset loaders for EndoVis17 and EndoVis18.
- Build a segmentation-to-tip localization pipeline for still images and frame sequences.
- Implement four deep-learning segmentation backends behind one unified inference interface.
- Build a temporal tracker for tip positions in video.
- Save tracking results as visual overlays and coordinate files.

### 3.2 Secondary goals

- Make the segmentation module replaceable.
- Support lightweight post-processing and temporal filtering.
- Provide evaluation scripts for both speed and localization quality.
- Allow side-by-side comparison across segmentation backends using the same dataset and visualization tooling.

## 4. Non-goals

- Editing or restructuring original dataset files
- Building a large-scale distributed training system
- Delivering a production-grade clinical GUI
- Supporting every dataset in the repository in the first implementation

## 5. User Scenarios

### Researcher

- Run segmentation-based tip tracking experiments on EndoVis sequences.
- Inspect frame-wise tip coordinates and visual outputs.
- Compare different segmentation backbones and post-processing settings.

### Developer

- Plug in a new segmentation model with minimal code changes.
- Extend the system with new dataset adapters.
- Reuse cached outputs under `temp/` for fast iteration.

## 6. Functional Requirements

### 6.1 Data I/O

- Provide dataset adapters that handle the different folder structures of EndoVis17 and EndoVis18.
- Read images, masks, and sequence metadata through a common API.
- Treat dataset folders as read-only inputs.
- Store all generated outputs under `temp/` or a user-specified output path.

### 6.2 Segmentation

- Support binary or multi-class instrument segmentation outputs.
- Accept either online model inference or precomputed segmentation masks.
- Support both single-frame and batch inference workflows.
- Expose a unified segmenter interface regardless of model family.
- Document backend-specific checkpoint discovery and pretrained-weight behavior.
- Provide a CLI path for running one backend on one image, one split, or one sequence.
- Save segmentation outputs, manifests, and overlays under `temp/runs/`.
- Support the following model families in the current roadmap:
  - ViT + CNN adapter style supervised model
  - MATIS-style masked-attention transformer model
  - SurgSAM-2 style surgical video segmentation model
  - SAM2 zero-shot promptable segmentation mode

### 6.3 Tip Localization

- Extract the most likely instrument region from a segmentation mask.
- Estimate tip candidates using contour analysis, connected components, skeletons, and principal axes.
- Handle multiple instruments through priority rules or instance separation.

### 6.4 Temporal Tracking

- Use previous tip states to reduce search regions or reorder candidates.
- Support smoothing with a Kalman filter or exponential moving average.
- Recover from short segmentation failures using temporal history.

### 6.5 Visualization and Export

- Overlay segmentation masks and tip markers on source frames.
- Export frame-wise coordinates as CSV or JSON.
- Export result videos or image sequences for inspection.

### 6.6 Evaluation

- Compute frame-wise distance error when ground-truth tip annotations are available.
- Define fallback metrics when only segmentation or tracking annotations are partially available.
- Measure FPS, average inference time, and average post-processing time.

## 7. Non-functional Requirements

- Implement the project in Python.
- Keep the codebase centered around the `esis/` package.
- Preserve dataset integrity by using a strict read-only input policy.
- Make experiments reproducible through clear configuration and CLI entry points.
- Support initial execution on CPU or a single GPU.
- Keep post-processing and tracking lightweight for fast iteration.

## 8. Dataset Plan

### EndoVis17

- Uses `train` and `val1` to `val10` splits.
- Provides paired `image` and `label` folders suitable for segmentation experiments.

### EndoVis18

- Should be supported through the same adapter-oriented design used for the earlier EndoVis datasets.
- Exact folder and annotation handling should be documented when the dataset parser is implemented.

### Excluded for now

- EndoVis15 and EndoVis19 are intentionally excluded from the current implementation scope.
- Their adapters may remain in the repository for future work, but active development and validation should target EndoVis17 and EndoVis18.

### Dataset handling rules

- Never modify source images, labels, or zip files.
- Save derived caches to `temp/cache/...`.
- Save generated indexes or metadata tables to `temp/index/...`.

## 9. Proposed Architecture

Expected project structure:

```text
esis/
  datasets/
    endovis17.py
    endovis18.py
    registry.py
  segmentation/
    base.py
    classical.py
    adapter_vit_cnn.py
    matis.py
    model_wrapper.py
    sam2_zero_shot.py
    surgsam2.py
    preprocessing.py
    postprocessing.py
  tracking/
    tip_detector.py
    temporal_filter.py
    tracker.py
  visualization/
    overlay.py
    video_writer.py
  evaluation/
    metrics.py
    benchmark.py
  utils/
    io.py
    geometry.py
    config.py
  cli/
    run_inference.py
    run_benchmark.py
temp/
  cache/
  runs/
  debug/
```

Pipeline:

1. A dataset loader reads frames and metadata.
2. A segmentation module generates or loads instrument masks through a unified segmenter API.
3. A tip detector estimates tip candidates from geometry.
4. A tracker combines current evidence with previous frame history.
5. Visualization and evaluation modules save outputs and compute metrics.

## 10. Implementation Phases

### Phase 1. Project skeleton

- Write `README.md` and `PRD.md`
- Create the `esis/` package structure
- Define `temp/` output conventions
- Design common configuration and CLI entry points

### Phase 2. Dataset adapters

- Implement EndoVis17 indexing and loading
- Implement EndoVis18 indexing and loading
- Validate sample frame loading and metadata parsing

### Phase 3. Segmentation backends and tip detection

- Finalize the shared segmentation interface
- Keep the existing mask-loader baseline for verification
- Implement the `adapter_vit_cnn` backend
- Implement the `matis` backend
- Implement the `surgsam2` backend
- Implement the `sam2_zero_shot` backend
- Add backend-specific wrappers while preserving one common call pattern
- Extract the instrument region with connected components
- Estimate tip position from contour extremity and skeleton analysis

### Phase 4. Temporal tracking

- Add frame-to-frame association
- Add ROI-based acceleration
- Add smoothing and fallback logic

### Phase 5. Evaluation and visualization

- Save experiment outputs
- Measure speed and localization quality
- Generate qualitative overlays and videos

### Segmentation execution notes

- Task-specific checkpoints should be stored under `temp/cache/checkpoints/`.
- Backend runners should resolve checkpoints by convention before requiring explicit paths.
- Segmentation CLI runs should write masks, overlays, and metadata into `temp/runs/segment/...`.

## 11. Success Criteria

- The system outputs tip coordinates on sample sequences from EndoVis17 and EndoVis18.
- All four deep-learning segmentation backends can be invoked through one consistent API.
- Tracking results can be visualized as overlays or videos.
- The pipeline runs fast enough for iterative experiments.
- All experiments are reproducible without changing original datasets.

## 12. Risks and Mitigations

- Poor segmentation quality may cause unstable tip estimates.
  - Mitigation: combine contour-based and skeleton-based candidates.
- Annotation formats differ across datasets.
  - Mitigation: isolate dataset-specific logic behind adapter layers.
- Different segmentation backends may require different prompt, preprocessing, or checkpoint conventions.
  - Mitigation: hide backend-specific setup behind a unified wrapper layer and standardized config schema.
- Real-time performance may be difficult.
  - Mitigation: use ROI cropping, frame skipping, lightweight models, and efficient post-processing.

## 13. Initial Technology Stack

- Python 3.13+
- NumPy
- OpenCV
- PyTorch
- torchvision
- scikit-image
- SciPy
- pandas
- tqdm
- matplotlib

## 14. Next Priorities

1. Create the `esis/` package and CLI skeleton.
2. Implement EndoVis17 and EndoVis18 dataset loaders.
3. Implement the shared segmentation interface and the four deep-learning segmentation backends.
4. Add tracking and visualization.
5. Add benchmarking and experiment automation.
