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
  - `pytorch/`
  - `PI_DeepOnet_Base_Analyse/`
  - `Classical_Solver/`
  - `Jax_Vs_Pytorch_Comparison/Multifamily/`
  - `Jax_Vs_Pytorch_Comparison/Monofamily/`

Migration note:
- The historical root-level folders (`src/`, `src_jax/`, `configs/`, `configs_jax/`, `launch/`, `benchmarks/`) are still present for backward compatibility.
- The new subtrees are the intended human-facing organization for navigation and documentation.
