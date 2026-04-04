# Physics-Informed DeepONets for the 1D ADR Equation

[![CI](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml/badge.svg)](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml)

This repository contains two closely related but distinct tracks:

1. `base/`: the canonical PyTorch implementation of the ADR PI-DeepONet.
2. `jax_comparison/`: the experimental comparison layer used to compare PyTorch and JAX under controlled protocols.

The repository is being cleaned so those two tracks are explicit instead of mixed implicitly through legacy top-level folders.

## Intended Branching Model

The recommended git organization is:

- `base`: stable branch for the canonical PyTorch ADR pipeline.
- `jax-comparison`: branch built on top of `base`, containing the JAX implementation and all comparison-specific material.

Current `main` still contains both tracks because it is the integration branch used during the cleanup.

Practical rule:

- changes that improve the canonical solver, data generation, training loop, or analysis belong to `base`
- changes that only exist to compare PyTorch and JAX belong to `jax-comparison`

## Repository Map

- [base/](/Users/emma.grospellier/Thèse/Projet_These_ADR/base): canonical PyTorch ADR workflow
- [jax_comparison/](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_comparison): comparison workspace layered on top of the base workflow
- [experiments/](/Users/emma.grospellier/Thèse/Projet_These_ADR/experiments): experiment-oriented organization of configs and launchers
- [benchmarks/](/Users/emma.grospellier/Thèse/Projet_These_ADR/benchmarks): benchmark helpers and shared benchmark configs
- [plot/](/Users/emma.grospellier/Thèse/Projet_These_ADR/plot): generated figures and summaries
- `results/`: runtime outputs used by analyses

Legacy top-level folders such as `src/`, `src_jax/`, `configs/`, `configs_jax/`, `launch/`, and `scripts/` remain active because some training and benchmark entry points still depend on them directly.

## Installation

### Base PyTorch environment

Use this for the canonical ADR pipeline:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### JAX comparison environment

Use this on top of the base environment for local CPU experiments:

```bash
pip install -r requirements-jax.txt
```

For GPU environments, especially HPC systems, do not assume `requirements-jax.txt` is sufficient. Install the platform-specific `jax` and `jaxlib` wheels first, then install the remaining comparison dependencies from `requirements-jax.txt`.

Jean Zay example:

- use the cluster-provided CUDA stack
- install the matching archived `jaxlib` wheel explicitly
- then install `optax`, `scipy`, `pyyaml`, `tqdm`, and plotting dependencies

## Documentation Entry Points

- [base/README.md](/Users/emma.grospellier/Thèse/Projet_These_ADR/base/README.md): canonical PyTorch workflow, scope, and entry points
- [jax_comparison/README.md](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_comparison/README.md): overall comparison workspace
- [jax_comparison/multifamily/README.md](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_comparison/multifamily/README.md): strict full-task comparison
- [jax_comparison/monofamily/README.md](/Users/emma.grospellier/Thèse/Projet_These_ADR/jax_comparison/monofamily/README.md): mono-family diagnostics and ablations
- [experiments/README.md](/Users/emma.grospellier/Thèse/Projet_These_ADR/experiments/README.md): target organization for configs and launchers

## What Is Stable vs Experimental

Stable:

- PyTorch ADR model and training pipeline
- canonical configs and launchers for the PyTorch workflow
- base analyses used for the main ADR conclusions

Experimental:

- JAX implementation under `src_jax/`
- benchmark harness under `benchmarks/`
- equal-pipeline PyTorch vs JAX benchmark material
- mono-family diagnostics
- ansatz and LBFGS ablations

## Cleanup Direction

The target end state is:

- `base/` remains the reference package for the stable PyTorch solver
- `jax_comparison/` remains a clear add-on package for framework comparison
- `experiments/` becomes the single place for human-facing experiment definitions
- legacy duplicated configs and launchers are removed once all active paths point to the cleaned layout
