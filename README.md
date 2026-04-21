# Physics-Informed DeepONets for Generalizable ADR Solvers

[![CI](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml/badge.svg)](https://github.com/Emma-Grspl/Physics-Informed-Deep-Operator-Networks-for-Generalizable-ADR-Solvers/actions/workflows/ci.yml)

This repository studies a physics-informed operator-learning approach for a one-dimensional advection-diffusion-reaction problem with parametric initial conditions. The work has two complementary goals: first, to build a reliable PI-DeepONet surrogate for the ADR equation; second, to compare PyTorch and JAX under matched training and evaluation protocols.

## Introduction

The physical problem considered in this repository is the one-dimensional advection-diffusion-reaction equation

\[
u_t + v\,u_x - D\,u_{xx} = \mu (u-u^3).
\]

The unknown field \(u(x,t)\) evolves under the combined effect of three mechanisms.

- The advection term \(v\,u_x\) models transport driven by a velocity field \(v\).
- The diffusion term \(D\,u_{xx}\) models spatial smoothing controlled by the diffusion coefficient \(D\).
- The reaction term \(\mu (u-u^3)\) introduces a local nonlinear source term controlled by \(\mu\).

This class of equation is representative of transport-reaction systems in which a quantity is simultaneously transported, diffused, and transformed locally. In this project, the goal is not to solve one isolated trajectory, but to learn a surrogate that generalizes over a family of ADR problems parameterized by physical coefficients and by the shape of the initial condition.

## Repository Skeleton

The `main` branch is the integration branch of the project. It is organized so that code, figures, saved models, and high-level assets are separated clearly.

- `base/`: canonical PyTorch baseline study
- `jax_vs_pytorch/`: strict framework-comparison study and related ablations
- `figures/`: curated scientific figures grouped by theme
- `assets/`: a short selection of the most representative visuals for quick inspection
- `models/`: root-level reference models exposed at repository level
- `Project_structure.md`: detailed map of the repository structure

The project also uses three main branches with distinct roles.

- `main`: global presentation branch, meant to expose the full project clearly
- `base`: branch focused on the PyTorch baseline only
- `jax-comparison`: branch focused on the PyTorch vs JAX comparison only

Within `main`, each scientific block has its own internal structure.

- `base/code/`: baseline PyTorch code, configs, launchers, and tests
- `base/README.md`: baseline-specific scientific summary
- `jax_vs_pytorch/code/`: JAX code, benchmark runners, comparison protocols, and experiment code
- `jax_vs_pytorch/figures/`: multifamily and monofamily comparison figures
- `jax_vs_pytorch/models/`: benchmark checkpoints and serialized trained parameters

## Parametric ADR Problem

The project studies a parametric ADR problem rather than one fixed equation instance. The surrogate must generalize across variations of the physical parameters and of the initial condition family.

The physical coefficients are sampled within the following ranges.

- \(v \in [0.5, 1.0]\)
- \(D \in [0.01, 0.2]\)
- \(\mu \in [0.0, 1.0]\)

The initial conditions are also parameterized.

- \(A \in [0.7, 1.0]\)
- \(\sigma \in [0.4, 0.8]\)
- \(k \in [1.0, 3.0]\)
- \(x_0 = 0\)

The spatial domain is

\[
x \in [-5, 8],
\]

and the time horizon depends on the protocol.

- Baseline study: \(T_{\max} = 3.0\)
- PyTorch vs JAX comparison: \(T_{\max} = 1.0\)

The reference solver and the audits use a structured space-time discretization.

- Baseline audits: \(N_x = 500\), \(N_t = 200\)
- JAX vs PyTorch comparison: \(N_x = 400\), \(N_t = 200\)

Three main families of initial conditions are considered.

- `Tanh`
- `Sin-Gauss`
- `Gaussian`

This family-based setup is central to the repository because the goal is to learn an operator that remains accurate across qualitatively different regimes, not simply to overfit one waveform.

## Reference Numerical Solver: Crank-Nicolson

All neural results are evaluated against a classical Crank-Nicolson reference solver. This choice is important for two reasons.

First, Crank-Nicolson is a standard implicit finite-difference method that offers a good compromise between accuracy and stability for time-dependent PDEs. It provides a trustworthy numerical target for evaluating the surrogate on the ADR problem.

Second, using a classical solver makes the scientific interpretation clearer. The neural model is not judged against its own training loss only, but against a numerical solution obtained by a known and interpretable reference method. This allows the repository to report relative \(L^2\) errors, inference speedups, and family-wise generalization results in a way that remains physically meaningful.

In this project, Crank-Nicolson therefore plays a dual role.

- It provides the reference solution used for benchmarking.
- It defines the baseline against which the surrogate acceleration is measured.

## Neural Network Description

The neural surrogate is a physics-informed Deep Operator Network designed for the ADR problem. Its role is to learn an operator that maps physical parameters, initial-condition parameters, and a query point \((x,t)\) to the solution value \(u(x,t)\).

The model uses a branch-trunk decomposition.

- The branch network encodes the ADR parameters and the initial-condition descriptors.
- The trunk network encodes the space-time query coordinates.
- The two representations are fused through conditional modulation layers inspired by FiLM-style interactions.

The branch input has dimension 8.

- \(v\)
- \(D\)
- \(\mu\)
- the family type identifier
- \(A\)
- \(x_0\)
- \(\sigma\)
- \(k\)

The trunk input has dimension 2.

- \(x\)
- \(t\)

The architecture used in both the baseline and the PyTorch vs JAX comparison is intentionally matched.

- branch depth: 5
- trunk depth: 4
- branch width: 256
- trunk width: 256
- latent dimension: 256
- multiscale Fourier features: 20
- Fourier scales: \(0, 1, 2, 3, 4, 5, 6, 8, 10, 12\)
- activation: SiLU

The trunk is enriched with multiscale Fourier features in order to better represent oscillatory or localized structures. This is particularly useful for the `Sin-Gauss` and `Gaussian` families, which are harder than the `Tanh` family.

The training objective combines three components.

- PDE residual loss
- initial-condition loss
- boundary-condition loss

The optimizer used during the main training phases is Adam. A final L-BFGS finisher is used in some protocols, especially in the baseline and in dedicated ablations, to test whether second-order polishing improves convergence. In practice, the experiments in this repository show that L-BFGS is not the dominant factor of improvement compared with architectural or ansatz-related choices.

The baseline learning rate is approximately \(6.08 \times 10^{-5}\). The training code also includes an explicit learning-rate decay strategy during the windowed optimization phases, with a progressive reduction toward small final values during long runs.

## Experimental Protocol for the Baseline Model

The baseline study answers the first scientific question of the repository: can a PI-DeepONet learn a reliable surrogate for the parametric ADR problem?

The baseline training protocol is more elaborate than a standard end-to-end run. It is built as a progressive time-marching strategy designed to stabilize optimization and to preserve solution quality over long horizons.

The main ingredients of the baseline protocol are the following.

- a warmup phase on the initial condition
- a progressive temporal curriculum
- a king-of-the-hill selection mechanism that keeps the best state seen so far
- rollback and retry logic when a training window does not validate
- adaptive PDE weighting through NTK-inspired balancing
- targeted correction on hard families detected during audit
- a final polishing phase with stricter validation

The baseline time curriculum is defined by three temporal zones.

- from \(t=0\) to \(t=0.05\): step size \(0.01\)
- from \(t=0.05\) to \(t=0.30\): step size \(0.05\)
- from \(t=0.30\) to \(t=3.0\): step size \(0.10\)

The main baseline hyperparameters are:

- batch size: 8192
- sampled training points per batch generation: 12288
- warmup iterations: 7000
- iterations per time window: 8000
- correction iterations: 9000
- number of outer loops: 3
- rolling window: 2000
- maximum retries: 4

The initial and final loss weights are also part of the protocol design.

- initial condition weight at start: 2000
- initial condition weight in final regime: 100
- boundary-condition weight: 200
- initial PDE weight: 500

Validation relies on explicit audit criteria.

- initial-condition threshold: 0.02
- time-step threshold: 0.03

The rationale behind this protocol is straightforward: the repository does not assume that a single monolithic optimization pass is sufficient to learn a stable ADR operator on a long horizon. Instead, it uses audits and corrective stages to enforce a scientific notion of convergence based on solution quality, not on training loss alone.

## Specific Protocol for PyTorch vs JAX

The comparison study addresses a different question: if the architecture and the physical task are held as constant as possible, which framework provides the better outcome for this ADR surrogate problem?

To answer that question, the PyTorch and JAX protocols are matched along the following dimensions.

- same geometry
- same parameter ranges
- same model width, depth, and latent dimension
- same multiscale Fourier encoding
- same branch/trunk logic
- same training horizon \(T_{\max}=1.0\)
- same audit grid and same evaluation families

The comparison protocol is intentionally shorter and stricter than the baseline study.

- \(T_{\max} = 1.0\)
- batch size: 4096
- sampled training points: 4096
- warmup iterations: 5000
- iterations per time window: 2500
- correction iterations: 4000
- number of outer loops: 1
- maximum retries: 2
- global audit cases: 40
- family audit cases: 12

The time curriculum for the comparison study is:

- from \(t=0\) to \(t=0.30\): step size \(0.05\)
- from \(t=0.30\) to \(t=1.0\): step size \(0.10\)

The benchmark evaluation protocol uses:

- 20 test cases per family
- three reported families: `Tanh`, `Sin-Gauss`, `Gaussian`
- seed 0 for the main public runs

The comparison retains several metrics because no single metric is sufficient.

- global relative \(L^2\) error
- family-wise relative \(L^2\) error
- full-grid inference time
- time-jump inference time
- speedup versus the Crank-Nicolson reference
- total training time

This design makes it possible to separate two notions that are often conflated.

- raw computational efficiency
- final scientific quality of the learned surrogate

The repository also includes more focused comparison studies.

- monofamily runs
- ansatz runs
- Gaussian-hypothesis ablations

These auxiliary experiments are used to diagnose where the difficulty comes from and whether the framework gap reflects a global issue or a family-specific failure mode.

## Numerical Results for the Baseline Model

The baseline conclusion is positive: the PI-DeepONet implemented in PyTorch learns a high-quality surrogate for the ADR problem under the multifamily protocol.

On the main multifamily benchmark with 20 evaluation cases per family, the baseline reaches:

- global relative \(L^2\): `0.00507 ± 0.00392`
- `Tanh`: `0.00139 ± 0.00035`
- `Sin-Gauss`: `0.00978 ± 0.00286`
- `Gaussian`: `0.00405 ± 0.00100`

The inference benchmark also shows a clear practical advantage over the reference solver.

- full-grid inference time: `0.210 s`
- time-jump inference time: `0.00285 s`
- Crank-Nicolson reference time: `0.499 s`
- speedup on the time-jump metric: `×175.03`

The full baseline benchmark training time for this short multifamily protocol is:

- total training time: `5329.21 s`

Scientific interpretation:

- the model is accurate enough to be used as a surrogate in the tested regime
- the surrogate is substantially faster than the reference solver at inference time
- the multifamily setup remains challenging, but the remaining error is concentrated mostly in the `Sin-Gauss` family

This is the main result supporting the use of the baseline branch as the scientific reference implementation of the project.

## Numerical Results for JAX

The JAX study must be interpreted in the context of direct comparison with PyTorch rather than in isolation. The most important result is that JAX is much faster in raw training time, but significantly worse in final solution quality under the matched protocol used here.

On the strict multifamily comparison benchmark, JAX reaches:

- global relative \(L^2\): `1.66884 ± 1.62812`
- `Tanh`: `1.23642 ± 0.15997`
- `Sin-Gauss`: `2.63937 ± 2.54170`
- `Gaussian`: `1.13073 ± 0.21905`

The timing results are:

- full-grid inference time: `0.249 s`
- time-jump inference time: `0.01079 s`
- Crank-Nicolson reference time: `0.490 s`
- speedup on the time-jump metric: `×45.38`
- total training time: `349.13 s`

The comparison therefore yields a strong asymmetry.

- PyTorch is far slower to train, but produces the reliable surrogate
- JAX is far faster to train, but does not match the required accuracy level in the main multifamily experiment

The monofamily results refine that conclusion.

PyTorch monofamily runs:

- `Tanh` only: `0.00158 ± 0.00048`
- `Sin-Gauss` only: `1.00000 ± 0.00000`
- `Gaussian` only: `0.87212 ± 0.01769`

JAX monofamily runs:

- `Tanh` only: `9.07159 ± 15.67342`
- `Sin-Gauss` only: `1.39526 ± 0.41086`
- `Gaussian` only: `1.02204 ± 0.05012`

These monofamily experiments show that the gap is not only a multifamily generalization issue. It is also driven by the difficulty of learning certain families, especially `Sin-Gauss` and `Gaussian`, and by the fact that JAX under the current training setup fails even on some reduced cases.

The ansatz experiments provide an additional diagnostic.

PyTorch ansatz runs:

- `Sin-Gauss` ansatz: `0.86448 ± 0.13245`
- `Gaussian` ansatz: `0.07643 ± 0.02998`

JAX ansatz run available in the public benchmark outputs:

- `Sin-Gauss` ansatz: `0.89959 ± 0.13354`

The Gaussian-hypothesis ablation confirms that the dominant improvement comes from imposing more structure on the initial condition rather than from using a more sophisticated final optimizer. Aggregated over three PyTorch seeds, the mean global relative \(L^2\) errors are:

- free learning, no L-BFGS: `0.82391 ± 0.06112`
- free learning, with L-BFGS: `0.86581 ± 0.07449`
- ansatz, no L-BFGS: `0.16063 ± 0.08412`
- ansatz, with L-BFGS: `0.21143 ± 0.13352`

This ablation supports two conclusions.

- The ansatz is the dominant lever for the Gaussian family.
- L-BFGS does not produce a robust gain in the tested setting and can even degrade the final result.

## Overall Assessment

The repository supports three main conclusions.

First, the baseline PyTorch PI-DeepONet is a valid and useful surrogate for the parametric ADR problem. It achieves low relative error on the main multifamily benchmark and delivers a clear inference speedup over the classical solver.

Second, the PyTorch vs JAX comparison shows that training speed and final scientific quality should be evaluated separately. In this project, JAX is clearly faster, but PyTorch is the only framework that achieves the level of accuracy required for the main ADR conclusions.

Third, the family-wise and ansatz-based experiments show that the most difficult part of the problem is not uniform across the initial-condition space. The `Tanh` family is relatively easy, while `Sin-Gauss` and `Gaussian` are more demanding. The way the initial condition is represented inside the model has a stronger effect on performance than the presence of an L-BFGS finisher.

From the standpoint of scientific use, the baseline PyTorch branch is therefore the reference branch. From the standpoint of methodological analysis, the JAX comparison branch is useful because it exposes the trade-off between computational speed and final surrogate quality.

## Usage

### Installation

Create and activate a Python virtual environment first.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the baseline dependencies:

```bash
pip install -r requirements-base.txt
```

Install the JAX comparison dependencies on top of the baseline environment:

```bash
pip install -r requirements-jax.txt
```

For GPU environments, install the appropriate `jax` and `jaxlib` build for your platform before installing the rest of the JAX requirements.

### Typical Commands

Run the baseline PyTorch training:

```bash
python base/code/scripts/train.py
```

Run the baseline hyperparameter search:

```bash
python base/code/scripts/tune_optuna.py
```

Run the PyTorch benchmark comparison pipeline:

```bash
python jax_vs_pytorch/code/benchmarks/pytorch/train_fulltrainer_benchmark.py \
  --config jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1.yaml
```

Run the JAX benchmark comparison pipeline:

```bash
python jax_vs_pytorch/code/benchmarks/jax/train_fulltrainer_benchmark.py \
  --config jax_vs_pytorch/code/benchmarks/configs/benchmark_fulltrainer_t1.yaml
```

Generate curated comparison figures:

```bash
python jax_vs_pytorch/code/code_experiments/build_curated_figures.py
```

### How to Navigate the Repository

- If you want the stable scientific baseline, start with `base/`.
- If you want the framework comparison and the ablations, go to `jax_vs_pytorch/`.
- If you want a concise visual overview, inspect `assets/`.
- If you want a full folder-by-folder map, read `Project_structure.md`.
