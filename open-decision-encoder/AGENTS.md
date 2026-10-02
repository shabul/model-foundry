# Open Decision Encoder working rules

Read IMPLEMENTATION.md before changing schema, split policy, serialization or release behavior.

- Keep this package independent of the parent repository's MLX environment.
- Never use DecisionBench for training, synthetic examples or checkpoint selection.
- Preserve official test splits; group source states before expansion/augmentation.
- Keep metadata, targets, source names and latent synthetic factors out of model input.
- Record revisions, hashes, random seeds, limits and failed experiments honestly.
- Only one process may use MPS for this project at a time; use runtime.accelerator_lock.
- Run offline unit tests before real training. Real training requires the hardware and tiny-overfit gates.
- Do not silently truncate options or change the meaning of abstention.
- Do not report a checkpoint as calibrated unless temperature fitting completed on calibration-only data.
- Do not publish artifacts automatically. The explicit package_release.py --repo command is the upload entry point.
- Run `uv run pytest -q` and `uv run ruff check src scripts tests` after behavior changes.
