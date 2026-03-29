# Physics-Informed DeepONet for the ADR Equation

[![CI](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml/badge.svg)](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Framework](https://img.shields.io/badge/framework-PyTorch-ee4c2c)

Research code for training, evaluating, and benchmarking a Physics-Informed Deep Operator Network (PI-DeepONet) on the 1D Advection-Diffusion-Reaction (ADR) equation.

Unlike standard PINNs, the model learns an operator from physical and initial-condition parameters to the full spatio-temporal solution, enabling fast inference across unseen regimes.

## Highlights

- Operator-learning approach for the nonlinear 1D ADR equation.
- Physics-informed training with PDE, initial-condition, and boundary-condition losses.
- Time-marching curriculum for long-horizon stability.
- Fourier-feature trunk encoding for higher-frequency solution patterns.
- Benchmarking against a Crank-Nicolson reference solver.

## Main Results

The target PDE is:

$$u_t + v u_x = D u_{xx} + \mu u (1 - u)$$

Average relative L2 error over 1000 random configurations per initial-condition family:

| Metric | Tanh | Sin-Gauss | Gaussian |
| --- | --- | --- | --- |
| Mean L2 Error | ~0.3% | ~2.9% | ~1.4% |

Inference benchmark for 50 scenarios at the final time horizon:

- Crank-Nicolson: `0.726 s`
- PI-DeepONet direct time-jump: `0.034 s`
- Speedup: `21x`

## Visual Outputs

Representative outputs are now organized by workflow:

- base PyTorch showcase: [`base/assets_pytorch/`](base/assets_pytorch)
- multifamily JAX vs PyTorch comparison: [`jax_comparison/multifamily/assets_multifamily/`](jax_comparison/multifamily/assets_multifamily)
- monofamily JAX vs PyTorch comparison: [`jax_comparison/monofamily/assets_monofamily/`](jax_comparison/monofamily/assets_monofamily)

## Repository Structure

- `configs/`: YAML configuration for geometry, optimization, and physical ranges.
- `scripts/`: entry points for training and Optuna-based tuning.
- `src/models/`: PI-DeepONet architecture.
- `src/physics/`: PDE residual computation with autograd.
- `src/training/`: training loop, curriculum logic, auditing, and polishing.
- `src/utils/`: Crank-Nicolson solver, metrics, and Optuna helpers.
- `src/analyse/`: benchmarking and figure-generation scripts.
- `test/`: numerical and physics-consistency tests.
- `assets/`: empty root-level placeholder for future showcase selection.
- `models_saved/`: pretrained checkpoints.

## Quick Start

### Prerequisites

- Python `3.11+`
- PyTorch

### Installation

```bash
git clone https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers.git
cd Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Common Commands

```bash
make test
make check
make analysis
make benchmark
make train
```

Equivalent direct commands:

```bash
python -m pytest -q
python -m compileall scripts src test
python src/analyse/global_analyse_PI_DeepOnet_vs_CN.py
python src/analyse/inference.py
python scripts/train.py
```

## Method Overview

The model combines:

- A branch network that encodes physical and initial-condition parameters.
- A trunk network that encodes `(x, t)` coordinates.
- Multiscale Fourier features in the trunk to reduce spectral bias.
- A physics-informed objective mixing PDE, IC, and BC terms.

Training includes several stabilization strategies:

- Time-marching curriculum across increasing temporal windows.
- Dynamic reweighting of loss terms with an NTK-inspired heuristic.
- Hybrid optimization with Adam followed by L-BFGS.
- Checkpoint rollback through a "King of the Hill" best-state mechanism.
- Targeted correction on failing initial-condition families.

## Reproducibility

- Automated checks run in GitHub Actions on pushes and pull requests.
- `pytest` covers PDE-residual and Crank-Nicolson reference validations.
- `compileall` is used as a lightweight source-integrity check.
- Main experiment settings are centralized in [`configs/config_ADR.yaml`](configs/config_ADR.yaml).

## Author

Emma Grospellier  
PhD research project
