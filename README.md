# Physics-Informed DeepONets for the 1D ADR Equation

[![CI](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml/badge.svg)](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml)

This branch is the stable PyTorch baseline of the project.

It focuses on one question:

- can a physics-informed DeepONet learn a reliable surrogate for the one-dimensional advection-diffusion-reaction equation?

## Physical Problem

The target equation is

\[
u_t + v\,u_x - D\,u_{xx} = \mu (u-u^3),
\]

where:

- \(v\) is the advection velocity
- \(D\) is the diffusion coefficient
- \(\mu\) controls the nonlinear reaction term

The model is trained on a parametric ADR setting with varying physical coefficients and several initial-condition families.

## Scientific Goal Of This Branch

This `base` branch is meant to answer whether the PyTorch PI-DeepONet pipeline works as a stable scientific baseline.

It is therefore the branch to read if you want:

- the canonical PyTorch implementation
- the reference ADR training pipeline
- the stable baseline results
- the branch that best represents the main surrogate-learning work independently of framework comparison

The PyTorch-versus-JAX comparison belongs to a separate branch, `jax-comparison`.

## What This Branch Contains

### Main Scientific Entry Point

- [base/](base): canonical PyTorch ADR workflow

### Supporting Directories

- [benchmarks/](benchmarks): benchmark runners and evaluation helpers still present in the repository
- [experiments/](experiments): protocol registry and experiment notes
- [plot/](plot): figures and curated visual outputs
- `results/`: runtime outputs and stored run artifacts

### Legacy Compatibility Layer

Some top-level folders remain because parts of the training and evaluation stack still depend on them:

- `src/`
- `configs/`
- `launch/`
- `scripts/`

They are active runtime infrastructure, but the recommended human-facing entry point is still `base/`.

## Reading Order

Recommended reading order for this branch:

1. this root `README.md`
2. [base/README.md](base/README.md)
3. the main configs under `base/configs/`
4. the training scripts under `base/scripts/`
5. the saved plots and assets under `base/plots/` and `base/assets_pytorch/`

## PI-DeepONet Baseline

The baseline model is a physics-informed DeepONet trained to approximate the ADR solution operator.

The model learns a mapping from:

- physical parameters
- initial-condition parameters
- a space-time query point \((x,t)\)

to the solution value \(u(x,t)\).

The training objective combines:

- PDE residual loss
- initial-condition loss
- boundary-condition loss

The reference numerical target is a Crank-Nicolson solver.

## Main Baseline Result

The PyTorch PI-DeepONet is the stable scientific reference of the repository.

On the reference multifamily benchmark with 20 evaluation cases per family:

- global relative L2: `0.00507 +- 0.00392`
- Tanh: `0.00139 +- 0.00035`
- Sin-Gauss: `0.00978 +- 0.00286`
- Gaussian: `0.00405 +- 0.00100`

Interpretation:

- the surrogate is accurate on the target ADR task
- the baseline is scientifically usable
- inference is substantially cheaper than the reference numerical solver

## Installation

Use the base PyTorch environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This branch should remain usable without JAX-specific dependencies.

## Reproducibility

Typical artifacts in this branch include:

- saved checkpoints
- training metrics
- evaluation against the Crank-Nicolson reference
- inference timing summaries
- analysis figures

The protocol definitions associated with the baseline are documented under [experiments/](experiments), but the baseline scientific implementation itself lives under [base/](base).

## About The Other Markdown Files

Some other root-level `.md` files are internal maintenance notes related to repository cleanup and branch organization.

They are not the primary entry points for readers of the baseline branch.
