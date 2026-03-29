# JAX Comparison

Comparison workspace for PyTorch vs JAX experiments built on top of the ADR base pipeline.

Subtrees:
- `multifamily/`: strict comparison on the main multi-family problem.
- `monofamily/`: exploratory family-by-family comparison runs and focused diagnostics.

Design choice:
- These folders are organized as self-contained experiment packages with `src/`, `scripts/`, `configs/`, `launch/`, `tests/`, and `plots/`.
- The canonical baseline implementation remains under `base/`.
