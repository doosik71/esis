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

## Model Reference

The table below summarizes the four supported learning-based backends, their related papers, and the most relevant official weight sources we could verify.

| Backend           | Related paper                                                                                                                                                   | Official code                                                               | Paper / checkpoint links                                                                                                                                                                                                                                                                        | ESIS implementation note                                                                                                                                                                                                                                                                                  |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `adapter_vit_cnn` | [Enhancing surgical instrument segmentation: integrating vision transformer insights with adapter](https://doi.org/10.1007/s11548-024-03140-z), Int J CARS 2024 | [AdapterSIS](https://github.com/weimengmeng1999/AdapterSIS)                 | Paper PDF: [AdapterSIS PDF](https://yuezijie.github.io/publications/AdapterSIS.pdf)<br>Backbone pretrain used by the official repo: [DINOv2 ViT-L/14 checkpoint](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth)                                                | ESIS uses a lightweight local ViT + CNN decoder with `timm` `vit_small_patch14_dinov2`. I did not find a public official EndoVis fine-tuned AdapterSIS checkpoint in the upstream repo, so this backend expects user-provided task checkpoints under `temp/cache/checkpoints/adapter_vit_cnn/<dataset>/`. |
| `matis`           | [MATIS: Masked-Attention Transformers for Surgical Instrument Segmentation](https://arxiv.org/abs/2303.09514), ISBI 2023                                        | [BCV-Uniandes/MATIS](https://github.com/BCV-Uniandes/MATIS)                 | Official bundle with data and pretrained models: [MATIS.tar.gz](http://157.253.243.19/MATIS/MATIS.tar.gz)<br>Alternative mirror from the repo: [Google Drive](https://drive.google.com/file/d/1sbOazLT49raQhteieVsJLo5PcKO9LEKG/view?usp=sharing)                                               | The original MATIS codebase uses a Mask2Former-based baseline plus a temporal module. The current ESIS `matis` backend is a simplified local implementation and only reuses MATIS-oriented checkpoint naming / storage conventions.                                                                       |
| `sam2_zero_shot`  | [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), 2024                                                                          | [facebookresearch/sam2](https://github.com/facebookresearch/sam2)           | Hugging Face model used by ESIS: [`facebook/sam2.1-hiera-tiny`](https://huggingface.co/facebook/sam2.1-hiera-tiny)<br>Transformers docs: [SAM2 model docs](https://huggingface.co/docs/transformers/en/model_doc/sam2)                                                                          | ESIS loads `Sam2Model` and `Sam2Processor` from Hugging Face and runs prompt-based zero-shot segmentation. No task-specific fine-tuned checkpoint is required for this backend.                                                                                                                           |
| `surgsam2`        | [Surgical SAM 2: Real-time Segment Anything in Surgical Video by Efficient Frame Pruning](https://arxiv.org/abs/2408.07931), NeurIPS 2024 Workshop AIM-FM       | [jinlab-imvr/Surgical-SAM-2](https://github.com/jinlab-imvr/Surgical-SAM-2) | Official fine-tuned EndoVis18 checkpoint from the repo README: [sam2.1_hiera_s_endo18.pth](https://drive.google.com/file/d/1DyrrLKst1ZQwkgKM7BWCCwLxSXAgOcMI/view?usp=drive_link)<br>Official SAM2 base family: [facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) | ESIS vendors the upstream Surgical-SAM-2 repository, but the current runtime path uses `SAM2ImagePredictor.from_pretrained(...)` with optional temporal mask reuse. That makes it a pragmatic prompt-driven approximation, not a full reproduction of the paper's training / evaluation pipeline.         |

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
uv run python main.py dataset gui
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
