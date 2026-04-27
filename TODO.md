# TODO

## Current Status

Completed items:

- Core project documents are present: `README.md`, `PRD.md`, `AGENT.md`, `SCHEDULE.md`
- The `esis/` package skeleton and `temp/` directory structure are in place
- Active dataset support is implemented for:
  - `data/EndoVis17 Instrument Subchallenge Dataset`
  - `data/EndoVis18 Instrument Subchallenge Dataset`
- Dataset indexing, inspection, validation, and debug preview export are implemented
- Dataset preview GUI is implemented
- Segmentation backends are wired behind one interface:
  - `mask_loader`
  - `adapter_vit_cnn`
  - `matis`
  - `sam2_zero_shot`
  - `surgsam2`
- Deep-learning runtime dependencies are configured in `pyproject.toml`
- Vendor research code has been cached under:
  - `temp/cache/vendors/AdapterSIS`
  - `temp/cache/vendors/MATIS`
  - `temp/cache/vendors/Surgical-SAM-2`

## Highest Priority Next

- Load real fine-tuned checkpoints for `adapter_vit_cnn`
- Load real fine-tuned checkpoints for `matis`
- Define a checkpoint directory convention under `temp/cache/checkpoints/`
- Document how each backend finds its checkpoint or pretrained weights
- Add a segmentation CLI command for running one backend on one image, one split, or one sequence
- Save segmentation outputs under `temp/runs/`

## Segmentation Follow-up

- Add backend load status reporting to the GUI
- Show model name, checkpoint path, and device in the GUI
- Add segmentation overlay export for selected samples
- Add per-backend runtime measurement
- Add per-backend mask quality comparison utilities
- Add clearer error messages when weights or vendor code are missing

## Tip Localization

- Implement `esis/tracking/tip_detector.py`
- Add connected-component filtering for candidate region selection
- Add contour extraction and contour-based tip candidates
- Add skeleton-based tip candidates
- Add tip scoring and tie-breaking rules
- Add multi-instrument handling rules

## Temporal Tracking

- Implement `esis/tracking/tracker.py`
- Implement `esis/tracking/temporal_filter.py`
- Add per-sequence tracker state management
- Add ROI restriction using previous tip position
- Add smoothing and short-term recovery logic
- Add frame-to-frame failure handling

## Visualization And Export

- Implement mask + tip overlay utilities in `esis/visualization/`
- Export annotated frames to `temp/runs/`
- Export annotated videos when the input is a sequence or video
- Export tip coordinates to CSV
- Export tip coordinates to JSON

## Evaluation And Benchmarking

- Finalize `esis/evaluation/metrics.py`
- Finalize `esis/evaluation/benchmark.py`
- Add localization error metrics
- Add mask overlap metrics where useful
- Add runtime and FPS benchmarking
- Add benchmark summary reports

## Experiment Workflow

- Add config-file based experiment runs
- Add named run directories under `temp/runs/`
- Add presets for dataset and backend combinations
- Add structured logging for model, dataset, and metric settings

## Testing And Validation

- Add smoke tests for dataset adapters
- Add smoke tests for each segmentation backend
- Add tests for preprocessing and postprocessing helpers
- Add tests for tip detection logic
- Add tests for tracker state transitions

## Documentation Cleanup

- Update `README.md` with actual deep-learning backend notes
- Document which backends currently use official pretrained weights
- Document which backends still need task-specific fine-tuned checkpoints
- Add backend setup instructions for first-time environment setup
- Add troubleshooting notes for Windows, Hugging Face cache, and GUI runtime issues

## Optional Cleanup

- Remove or archive old debug outputs for excluded datasets if they are no longer needed
- Remove or archive stale index files for excluded datasets if they are no longer needed
- Recheck all docs so they consistently refer only to EndoVis17 and EndoVis18 as active scope
