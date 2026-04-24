# AGENT Guide

## Data Folder Policy

- Do not modify any files or folders under `data/`.
- Treat all dataset contents under `data/` as read-only shared assets.
- Do not rename, delete, overwrite, re-encode, or move dataset files inside `data/`.
- Do not create cache files, logs, checkpoints, indexes, or temporary outputs inside `data/`.
- Store all generated artifacts in locations such as `temp/`, not in `data/`.
- This policy applies to all target dataset paths, including:
  - `data/EndoVis15 Instrument Subchallenge Dataset`
  - `data/EndoVis17 Instrument Subchallenge Dataset`
  - `data/EndoVis18 Instrument Subchallenge Dataset`
  - `data/EndoVis19 Instrument Subchallenge Dataset`

## Package Management Policy

- Use `uv` as the default Python package manager for this project.
- Prefer `uv add <package>` when adding dependencies to the project.
- Prefer `uv remove <package>` when removing dependencies.
- Prefer `uv sync` to install project dependencies from `pyproject.toml` and lock data.
- Prefer `uv run <command>` to run Python scripts and tools inside the project environment.
- Avoid using `pip install` directly unless there is a clear reason that `uv` cannot handle the task.

## Examples

- Add a package: `uv add opencv-python`
- Run the app: `uv run python main.py`
- Sync dependencies: `uv sync`
