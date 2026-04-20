# PyTorch vs JAX for Physics-Informed DeepONets on the 1D ADR Equation

[![CI](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml/badge.svg)](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml)

This branch is the comparison branch of the project.

Its role is not to ask whether the ADR PI-DeepONet works in absolute terms. That is the role of the `base` branch.

Its role is to answer a different question:

- when the same ADR operator-learning problem is trained under PyTorch and JAX workflows, which framework delivers the better scientific outcome?

This root README is intended to be self-sufficient. A recruiter, collaborator, or reviewer should be able to understand the comparison protocol, the main results, the ablations, the limitations, and the final conclusion without searching through the repository.

## Executive Summary

This branch compares PyTorch and JAX on a physics-informed DeepONet for the one-dimensional advection-diffusion-reaction equation.

The main conclusion is clear:

- JAX is much faster in raw training time
- PyTorch is much better in final solution quality on the real multifamily ADR task

The comparison becomes more informative when broken down:

- the multifamily benchmark provides the main framework conclusion
- monofamily benchmarks show that some families are intrinsically harder than others
- ansatz-based experiments show that initial-condition structure matters strongly
- the Gaussian Hypothesis ablation shows that ansatz matters much more than L-BFGS in this study

If someone only reads one sentence from this branch, it should be:

- for this ADR problem and this training logic, PyTorch is the reliable scientific choice, while JAX is the faster but substantially less accurate one

## Problem Setting

The physical problem is the one-dimensional advection-diffusion-reaction equation

\[
u_t + v\,u_x - D\,u_{xx} = \mu (u-u^3),
\]

where:

- \(v\) is the advection velocity
- \(D\) is the diffusion coefficient
- \(\mu\) controls the nonlinear reaction term

The task is parametric. The model must learn a surrogate over varying physical coefficients and varying initial conditions rather than solve a single fixed case.

The benchmarked families of initial conditions are:

- `Tanh`
- `Sin-Gauss`
- `Gaussian`

This makes the task a genuine operator-learning problem with heterogeneous difficulty across families.

## What This Branch Is Meant To Prove

This branch answers four practical questions:

1. which framework gives the best final ADR surrogate under comparable protocols?
2. how large is the speed advantage of JAX in wall-clock training?
3. do the same failure modes appear in multifamily and monofamily settings?
4. does adding an ansatz or an L-BFGS finisher materially change the conclusion?

It is therefore not a generic “JAX port” branch. It is a structured empirical comparison.

## Architecture And Training Logic

Both frameworks use the same modeling idea:

- a physics-informed DeepONet
- a branch network for physical and initial-condition parameters
- a trunk network for space-time coordinates
- a final interaction between both representations to predict \(u(x,t)\)

The comparison keeps the architecture family aligned:

- branch depth: `5`
- trunk depth: `4`
- branch width: `256`
- trunk width: `256`
- latent dimension: `256`
- Fourier features: `20`

The shared training philosophy is also aligned:

1. warm up on the initial condition
2. train progressively in time
3. audit the model against the Crank-Nicolson reference
4. identify hard regimes
5. apply targeted correction
6. optionally use a final refinement stage

The loss remains physics-informed in both frameworks and combines:

- PDE residual loss
- initial-condition loss
- boundary-condition loss

The point of the branch is not to compare arbitrary implementations. It is to compare two frameworks under a closely matched scientific protocol.

## Reference Comparison Protocol

The most important public comparison in this branch is the strict multifamily benchmark.

Its visible branch-level settings are:

- target horizon: `T_max = 1.0`
- batch size: `4096`
- sampled collocation points per draw: `4096`
- warmup iterations: `5000`
- iterations per time step: `2500`
- correction iterations: `4000`
- number of correction loops: `1`
- evaluation set: `20` cases per family
- benchmark seed: `0`

The benchmark compares:

- training time
- final relative L2 error
- family-wise error
- inference timing against Crank-Nicolson

The multifamily benchmark is the primary result.

The monofamily and ansatz experiments are diagnostic layers that explain why the main result looks the way it does.

## Main Results

### Multifamily Benchmark

This is the decisive result of the branch.

PyTorch on the strict three-family benchmark:

- global relative L2: `0.00507 +- 0.00392`
- Tanh: `0.00139 +- 0.00035`
- Sin-Gauss: `0.00978 +- 0.00286`
- Gaussian: `0.00405 +- 0.00100`
- training time: about `5329 s`
- time-jump speedup vs Crank-Nicolson: about `x175`

JAX on the matched strict three-family benchmark:

- global relative L2: `1.66884 +- 1.62812`
- Tanh: `1.23642 +- 0.15997`
- Sin-Gauss: `2.63937 +- 2.54170`
- Gaussian: `1.13073 +- 0.21905`
- training time: about `349 s`
- time-jump speedup vs Crank-Nicolson: about `x45`

Interpretation:

- JAX is dramatically faster in training time
- this gain does not translate into a scientifically competitive surrogate
- PyTorch is the only backend that delivers the main target result at a strong level of accuracy

This is the central conclusion of the entire branch.

## Monofamily Diagnostics

The monofamily experiments are not the main benchmark, but they are essential for interpretation.

PyTorch monofamily results:

- `Tanh` only: `0.00158 +- 0.00048`
- `Sin-Gauss` only: about `1.00000`
- `Gaussian` only: `0.87212 +- 0.01769`

JAX monofamily results:

- `Tanh` only: `9.07159 +- 15.67342`
- `Sin-Gauss` only: `1.39526 +- 0.41086`
- `Gaussian` only: `1.02204 +- 0.05012`

Interpretation:

- `Tanh` is easy for PyTorch but unstable for JAX in the tested setup
- `Sin-Gauss` is genuinely difficult for both frameworks
- `Gaussian` remains difficult when learned freely
- the framework gap is not only a multifamily effect

This matters because it shows that the comparison is not just “PyTorch wins because the joint task is hard”. Some families are already problematic in isolation.

## Ansatz Results

The branch also includes ansatz-oriented experiments designed to test whether explicitly structuring the initial condition improves learning.

PyTorch ansatz results:

- `Sin-Gauss` with ansatz: `0.86448 +- 0.13245`
- `Gaussian` with ansatz: `0.07643 +- 0.02998`

JAX ansatz results:

- `Sin-Gauss` with ansatz: `0.89959 +- 0.13354`

Interpretation:

- ansatz helps little on `Sin-Gauss`
- ansatz helps strongly on `Gaussian`
- the representation and enforcement of the initial condition is one of the main levers of performance in this repository

This is one of the most important messages to make visible in the branch, because it explains part of the difficulty in a mechanistic way rather than only reporting error numbers.

## Gaussian Hypothesis Ablation

The Gaussian Hypothesis ablation isolates two factors:

- free learning versus ansatz on the initial condition
- with and without an L-BFGS finisher

Aggregated results across three seeds:

- PyTorch free / no LBFGS: `0.8239 +- 0.0611`
- PyTorch free / LBFGS: `0.8658 +- 0.0745`
- PyTorch ansatz / no LBFGS: `0.1606 +- 0.0841`
- PyTorch ansatz / LBFGS: `0.2114 +- 0.1335`
- JAX free / no LBFGS: `1.0065 +- 0.0060`
- JAX free / LBFGS: `1.0065 +- 0.0059`
- JAX ansatz / no LBFGS: `0.4814 +- 0.0056`
- JAX ansatz / LBFGS: `0.4802 +- 0.0056`

Interpretation:

- the ansatz is the dominant helpful factor
- L-BFGS does not provide a robust gain in the tested Gaussian setting
- PyTorch remains clearly ahead in final error

This ablation sharpens the overall branch conclusion:

- the main bottleneck is not simply optimizer choice
- the main bottleneck is also how the initial condition is encoded into the learning problem

## What This Branch Establishes

This branch supports the following conclusions:

- framework choice materially changes the final quality of the learned ADR surrogate
- raw training speed is not a sufficient metric for scientific usefulness
- PyTorch is the reliable backend for the main ADR result in this repository
- JAX is valuable as a comparison and diagnostic backend, but not as the reference backend under the tested setup
- ansatz-based structure can matter more than second-order finishing in the hard Gaussian regime

## Limitations

The branch is useful because it is honest about what works and what does not.

Scientific limitations:

- the matched JAX workflow is not competitive in final quality in the current study
- some hard families remain difficult even after narrowing the task
- the branch provides strong empirical evidence, but not a theoretical proof of framework superiority

Repository limitations:

- the branch still sits inside a larger research repository rather than a perfectly minimal comparison package
- some infrastructure was inherited from iterative experimentation rather than designed top-down in one pass
- the content is now documented clearly, but the repository still reflects active research history

These limitations should be visible. They make the branch more credible, not less.

## What To Read In This Branch

If you want the quick branch-level story:

1. this README
2. the figures under `assets/`
3. the comparison subtree under `jax_comparison/`

If you want the protocol and reproduction layer:

1. `experiments/`
2. `benchmarks/`
3. `results/`

The intended rule is:

- this root README gives the full branch narrative
- the rest of the branch exists to support, reproduce, or inspect that narrative

## Environment

Install the base environment first:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then install the comparison-specific dependencies:

```bash
pip install -r requirements-jax.txt
```

For GPU systems and HPC environments, install the compatible `jax` and `jaxlib` build first, then install the remaining comparison dependencies.
