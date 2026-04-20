# PyTorch vs JAX for Physics-Informed DeepONets on the 1D ADR Equation

[![CI](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml/badge.svg)](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml)

This branch is the framework-comparison branch of the project.

Its purpose is not to present the canonical PyTorch baseline in isolation. Its purpose is to answer a narrower question:

- when training a physics-informed DeepONet on the one-dimensional advection-diffusion-reaction equation, is PyTorch or JAX the better framework in practice?

## Physical Problem

The target equation is

\[
u_t + v\,u_x - D\,u_{xx} = \mu (u-u^3),
\]

with parametric physical coefficients and several families of initial conditions.

This branch therefore studies framework behavior on a nontrivial operator-learning problem rather than on a toy scalar regression task.

## Scientific Goal Of This Branch

This `jax-comparison` branch exists to compare:

- final solution quality
- training speed
- inference speed
- stability across multifamily and monofamily settings
- the effect of ansatz-based constraints and L-BFGS finishing

The stable baseline branch is `base`. This branch should be read as the comparison and interpretation layer added on top of that baseline.

## What This Branch Contains

### Main Comparison Entry Points

- [jax_comparison/](jax_comparison): comparison-specific workspace
- [jax_comparison/multifamily/README.md](jax_comparison/multifamily/README.md): main strict full-task comparison
- [jax_comparison/monofamily/README.md](jax_comparison/monofamily/README.md): diagnostic studies and ablations

### Protocol Registry

- [experiments/](experiments): reproducible experiment definitions
- [experiments/multifamily/README.md](experiments/multifamily/README.md): main comparison protocols
- [experiments/monofamily/README.md](experiments/monofamily/README.md): monofamily protocols
- [experiments/ablations/gaussian_hypothesis/README.md](experiments/ablations/gaussian_hypothesis/README.md): Gaussian ansatz / LBFGS ablation

### Benchmark Execution Layer

- [benchmarks/](benchmarks): shared train / eval / inference runners for both frameworks

### Outputs And Assets

- `results/`: active benchmark outputs and run artifacts
- [plot/](plot): figure hub
- `assets/`: curated visual assets

## Reading Order

Recommended reading order on this branch:

1. this root `README.md`
2. [jax_comparison/README.md](jax_comparison/README.md)
3. [experiments/multifamily/README.md](experiments/multifamily/README.md)
4. [jax_comparison/multifamily/README.md](jax_comparison/multifamily/README.md)
5. [jax_comparison/monofamily/README.md](jax_comparison/monofamily/README.md)
6. [experiments/ablations/gaussian_hypothesis/README.md](experiments/ablations/gaussian_hypothesis/README.md)

If you only care about the main conclusion, go directly to the strict multifamily comparison.

## Main Comparison Conclusion

The central conclusion of this branch is:

- JAX is much faster in raw training time
- PyTorch is much better in final solution quality on the actual three-family ADR task

So in this project, PyTorch is the reliable framework for the main scientific result, while JAX is valuable mainly as a comparison target and as a tool for investigating optimization behavior.

## Main Results Snapshot

### Strict Multifamily Comparison

This is the principal benchmark of the branch.

PyTorch reference result on the strict multifamily task:

- global relative L2: `0.00507 +- 0.00392`
- Tanh: `0.00139 +- 0.00035`
- Sin-Gauss: `0.00978 +- 0.00286`
- Gaussian: `0.00405 +- 0.00100`

Interpretation:

- the PyTorch surrogate remains accurate on the real target task
- the matched JAX pipeline is faster, but not competitive in final error quality in this repository

### Monofamily Diagnostics

The monofamily studies exist to answer questions such as:

- is a family intrinsically difficult?
- does a hard family remain hard when isolated?
- does the difficulty come from multifamily generalization or from the family itself?

These results are explanatory, not the main scientific benchmark.

### Gaussian Hypothesis Ablation

The Gaussian ablation isolates two factors:

- free learning versus ansatz for the initial condition
- with and without an L-BFGS finisher

The main conclusion is:

- the ansatz is the dominant helpful factor
- L-BFGS does not provide a robust gain in the tested setting

## Branch Identity

This branch is comparison-first.

That means:

- changes motivated by framework comparison belong here
- benchmark infrastructure used to compare frameworks belongs here
- JAX implementation details belong here
- monofamily diagnostics and Gaussian ablations belong here

What does not define this branch:

- the stable PyTorch baseline in isolation

That stable baseline belongs conceptually to the `base` branch.

## Installation

Install the base PyTorch environment first:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then add the comparison-layer dependencies:

```bash
pip install -r requirements-jax.txt
```

For GPU machines and HPC systems, install the platform-compatible `jax` and `jaxlib` build first, then install the remaining dependencies.

## Reproducibility

This branch is organized around reproducible benchmark artifacts such as:

- training metrics
- saved checkpoints or serialized parameters
- evaluation against the Crank-Nicolson reference
- inference timing summaries
- benchmark summaries aggregated by backend and seed

The public protocol definitions live under [experiments/](experiments).

## About The Other Root Markdown Files

Other root-level `.md` files are internal maintenance notes related to cleanup and branch organization.

They are not the primary entry points for understanding the comparison results.
