# Project Structure

The repository is organized around three main entry points:

- `base/`
  - canonical PyTorch ADR implementation and base analysis workflow
- `jax_comparison/multifamily/`
  - strict PyTorch vs JAX comparison on the main multi-family task
- `jax_comparison/monofamily/`
  - exploratory mono-family comparison runs and focused diagnostics

Central plot hub:
- `plot/`
  - `Pytorch/`
  - `PI_DeepOnet_Base_Analyse/`
  - `Classical_Solver/`
  - `Jax_Vs_Pytorch_Comparison/Multifamily/`
  - `Jax_Vs_Pytorch_Comparison/Monofamily/`

Current navigation rule:
- `base/` is the human-facing home for the PyTorch baseline, Optuna tuning, saved model, and base plots.
- `jax_comparison/` is the human-facing home for all framework comparisons.
- `plot/` is the central visual gallery.

Legacy runtime note:
- The historical root-level folders (`src/`, `src_jax/`, `configs/`, `configs_jax/`, `launch/`, `benchmarks/`, `scripts/`, `scripts_jax/`, `test/`) are still present because several benchmark scripts still import them directly at runtime.
- They should now be treated as compatibility infrastructure, not as the main navigation entry points.
- A deeper refactor is still required before those root runtime folders can be safely removed.
