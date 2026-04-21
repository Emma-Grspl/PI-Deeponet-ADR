# JAX vs PyTorch Subtree

`jax_vs_pytorch/` is the comparison block exposed inside the `main` branch.

Its role is not to establish that the ADR PI-DeepONet works in absolute terms. That is the role of `base/`. This subtree exists to compare two closely matched workflows, one in PyTorch and one in JAX, and to expose the diagnostic experiments that explain the framework gap.

## Role

This subtree groups the comparison-specific material of the repository:

- benchmark runners for PyTorch and JAX
- JAX implementation code
- matched comparison configurations
- experiment registries for multifamily, monofamily, and ablation studies
- full figure sets and curated comparison assets
- serialized benchmark outputs and model artifacts

In other words:

- `base/` carries the stable baseline result
- `jax_vs_pytorch/` carries the framework-comparison result

## Structure

- `code/benchmarks/`
  Standardized runners for training, evaluation, and inference.
- `code/configs/`
  PyTorch-side comparison configs.
- `code/configs_jax/`
  JAX-side comparison configs.
- `code/src_jax/`
  JAX implementation of the ADR workflow.
- `code/experiments/`
  Registry of multifamily, monofamily, and ablation protocols.
- `code/code_experiments/`
  Plotting and synthesis scripts dedicated to the comparison studies.
- `code/launch/`
  SLURM launchers for the comparison workflows.
- `figures/`
  Full comparison figure sets.
- `assets/`
  Curated comparison visuals for quick reading.
- `models/`
  Serialized benchmark artifacts grouped by backend and protocol.

## Main Takeaway

The central scientific outcome surfaced by this subtree is:

- JAX is much faster in raw training time
- PyTorch is much better in final solution quality on the main multifamily ADR benchmark

The multifamily comparison establishes the headline result, while the monofamily and ablation studies explain where the difficulty comes from and why the framework gap appears.

The full scientific narrative remains at repository root in `README.md`. This subtree exists as the clean technical home for the comparison layer.
