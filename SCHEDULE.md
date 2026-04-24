# Development Schedule

## Overview

This document defines the sequential development schedule for the surgical instrument tip tracking project.

Target datasets:

- `data/EndoVis17 Instrument Subchallenge Dataset`
- `data/EndoVis18 Instrument Subchallenge Dataset`

Excluded from the current development phase:

- `data/EndoVis15 Instrument Subchallenge Dataset`
- `data/EndoVis19 Instrument Subchallenge Dataset`

Development rules:

- Do not modify anything under `data/`.
- Keep most Python modules under `esis/`.
- Save temporary outputs, caches, logs, and debug artifacts under `temp/`.
- Use `uv` as the default Python package manager.

## Phase 0. Documentation and Project Rules

### Goal

Lock down the project scope, rules, and baseline documentation before implementation starts.

### Tasks

- Finalize `README.md`
- Finalize `PRD.md`
- Finalize `AGENT.md`
- Write this `SCHEDULE.md`

### Deliverables

- Project overview document
- Product requirements document
- Repository working rules
- Sequential development schedule

### Exit criteria

- All core documents are present in the repository root.
- Dataset handling rules and package management rules are clearly documented.

## Phase 1. Project Skeleton Setup

### Goal

Create the base Python package structure and execution entry points.

### Tasks

- Create `esis/` package layout
- Add subpackages for datasets, segmentation, tracking, visualization, evaluation, utils, and cli
- Create `temp/` directory structure for cache, runs, debug, and indexes
- Update `main.py` to serve as a simple project entry point
- Prepare `pyproject.toml` for package and dependency management with `uv`

### Deliverables

- Importable `esis` package
- Basic CLI or runner entry point
- Temporary artifact directory conventions

### Exit criteria

- The repository has a clean package skeleton.
- `uv run python main.py` executes successfully.

## Phase 2. Dataset Adapter Foundation

### Goal

Implement a unified dataset interface for the active EndoVis17 and EndoVis18 datasets.

### Tasks

- Define a common dataset sample schema
- Implement dataset registry logic
- Implement `endovis17.py`
- Implement `endovis18.py`
- Add metadata parsing and sequence indexing
- Save generated indexes under `temp/index/`

### Deliverables

- Shared dataset interface
- Dataset-specific adapters
- Dataset indexing utilities

### Exit criteria

- Each dataset loader can enumerate available sequences and frames.
- No files inside `data/` are modified.
- Metadata caches are stored only under `temp/`.

## Phase 3. Data Validation and Debug Utilities

### Goal

Verify that dataset adapters are correct and easy to inspect.

### Tasks

- Add frame preview utilities
- Add dataset summary reporting
- Add shape, dtype, and annotation sanity checks
- Export debug previews to `temp/debug/`

### Deliverables

- Dataset inspection scripts
- Debug images and summary reports

### Exit criteria

- A developer can inspect representative samples from each dataset.
- Loader issues can be diagnosed without touching raw datasets.

## Phase 4. Baseline Segmentation Pipeline

### Goal

Build the first working segmentation stage that feeds the tip tracker.

### Tasks

- Define a segmentation interface
- Implement a simple classical baseline or mask-loader baseline
- Add model wrapper support for future learned backbones
- Add pre-processing and post-processing helpers

### Deliverables

- Segmentation base classes
- Baseline segmentation implementation
- Reusable segmentation pipeline entry point

### Exit criteria

- The system can generate or load instrument masks for sample frames.
- The segmentation module can be swapped without changing downstream code.

## Phase 5. Tip Localization

### Goal

Estimate instrument tip coordinates from segmentation masks.

### Tasks

- Implement connected component filtering
- Implement contour extraction
- Implement skeleton-based candidate generation
- Implement extremity and axis-based tip scoring
- Handle multi-instrument cases with a consistent rule set

### Deliverables

- `tip_detector.py`
- Geometry helper utilities
- Frame-wise tip coordinate outputs

### Exit criteria

- The system returns tip coordinates for sample images.
- Tip estimates are visually plausible on representative examples.

## Phase 6. Temporal Tracking

### Goal

Stabilize tip positions across frame sequences and videos.

### Tasks

- Implement tracker state management
- Add ROI-based search restriction
- Add temporal smoothing
- Add fallback logic for missing or noisy masks
- Export per-sequence tracking results

### Deliverables

- `tracker.py`
- `temporal_filter.py`
- Sequence-level coordinate outputs

### Exit criteria

- The system tracks tips continuously across sequences.
- Short segmentation failures do not immediately collapse tracking.

## Phase 7. Visualization and Result Export

### Goal

Make outputs easy to inspect and compare.

### Tasks

- Overlay masks and tip markers on frames
- Export frame sequences with annotations
- Export videos when possible
- Save coordinates in CSV or JSON format under `temp/runs/`

### Deliverables

- Visualization utilities
- Video and image-sequence export
- Coordinate result files

### Exit criteria

- A user can review both numeric and visual outputs from one experiment run.

## Phase 8. Evaluation and Benchmarking

### Goal

Measure quality and runtime performance in a repeatable way.

### Tasks

- Implement localization error metrics
- Implement runtime and FPS measurement
- Add experiment summary reporting
- Add benchmarking CLI commands

### Deliverables

- `metrics.py`
- `benchmark.py`
- Benchmark output reports

### Exit criteria

- Accuracy and speed can be measured from a standard command.
- Experiment comparisons are reproducible.

## Phase 9. Experiment Automation

### Goal

Make repeated experiments easier to run and compare.

### Tasks

- Add configuration file support
- Add run naming and output folder conventions
- Add experiment presets for dataset and model combinations
- Add logging for key settings and metrics

### Deliverables

- Config utilities
- Standardized run directories
- Reproducible experiment commands

### Exit criteria

- A user can rerun the same experiment with the same settings and output structure.

## Phase 10. Refinement

### Goal

Improve accuracy, speed, and code quality after the baseline is working.

### Tasks

- Tune pre-processing and post-processing
- Improve tip candidate ranking
- Optimize runtime bottlenecks
- Add tests for critical utility modules
- Improve documentation where implementation details became clearer

### Deliverables

- Improved performance
- Better code stability
- Updated technical documentation

### Exit criteria

- The baseline system is stable enough for repeated research experiments.
- Core modules have tests or validation scripts for the most important behaviors.

## Recommended Execution Order

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7
9. Phase 8
10. Phase 9
11. Phase 10

## Immediate Next Actions

1. Create the `esis/` package skeleton.
2. Create the `temp/` directory conventions.
3. Implement the dataset registry and dataset adapters.
4. Add a first end-to-end baseline from mask input to tip output.
