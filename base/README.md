# Base PyTorch Subtree

`base/` is the PyTorch baseline block exposed inside the `main` branch.

If a reader wants the canonical ADR surrogate without the comparison layer, this is the correct subtree to inspect first.

## Role

This subtree contains the reference PyTorch workflow used to support the main baseline conclusions of the project:

- parametric ADR data generation
- PI-DeepONet model definition
- PDE residual computation
- time-marching training
- comparison against the Crank-Nicolson reference solver
- baseline figures and curated assets

In other words:

- `base/` answers whether the PI-DeepONet works on the ADR task
- `jax_vs_pytorch/` answers how the JAX workflow compares to the PyTorch reference

## Structure

- `code/configs/`
  Canonical baseline configurations.
- `code/scripts/`
  Main training and tuning entry points.
- `code/src/`
  PyTorch implementation of the ADR surrogate pipeline.
- `code/launch/`
  SLURM launchers for the baseline workflow.
- `code/tests/`
  Small regression and numerical sanity checks.

This subtree is intentionally compact inside `main`: it surfaces the core baseline implementation without absorbing comparison-only material.

## What A Reader Should Take Away

Someone reading `base/` should be able to answer four questions quickly:

1. what physical problem is being solved?
2. what neural architecture is used?
3. what is the training logic?
4. what should be treated as the stable PyTorch reference result?

The detailed scientific presentation is at repository root in `README.md`. This subtree exists to provide a clean technical entry point.
