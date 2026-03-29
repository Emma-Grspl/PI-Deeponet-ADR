# Base

Canonical PyTorch ADR implementation used as the foundation of the repository.

Contents:
- `assets_pytorch/`: curated PyTorch-only assets kept near the base workflow.
- `src/`: main PyTorch solver, data generation, physics, training, and analysis code.
- `scripts/`: base training and tuning entry points.
- `configs/`: canonical PyTorch configs.
- `launch/`: generic SLURM launchers for the base workflow.
- `tests/`: reference tests for the classical solver and PDE residual.
- `plots/`: organized plot hub copies for PyTorch-only outputs and classical-solver plots.
- `models_saved/`: saved PyTorch checkpoint used by the base analysis.

Notes:
- This subtree is the reference implementation for the ADR operator-learning workflow.
- JAX benchmark and comparison material lives under `jax_comparison/`.
