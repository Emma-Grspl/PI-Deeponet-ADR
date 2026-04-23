# Experimental Protocol

## Overview

This repository follows a two-stage experimental strategy:

### Stage A — Scientific Validation (Steps 1–7)

Demonstrate that a Physics-Informed DeepONet (PI-DeepONet) can serve as a robust and accurate surrogate solver for a parametric 1D advection-diffusion-reaction (ADR) equation.

### Stage B — Framework Benchmarking (Steps 8–11)

Evaluate, under matched experimental conditions, how **PyTorch** and **JAX** compare in terms of:

- final scientific accuracy  
- robustness of convergence  
- training speed  
- inference speed  
- sensitivity to architectural priors and optimization choices

The benchmark conclusions are specific to this ADR problem and this training setup.

---

# Stage A — Scientific Validation of the PI-DeepONet

---

## Step 1 — Definition of the ADR Problem Setup

We first define the full family of ADR equations studied in this repository.

### Governing Equation

$$
u_t + v u_x - D u_{xx} = \mu (u - u^3)
$$

where:

- $\(v\)$: advection velocity  
- $\(D\)$: diffusion coefficient  
- $\(\mu\)$: nonlinear reaction strength

### Parameter Ranges

Physical coefficients are sampled within predefined ranges:

- $\(v \in [v_{min}, v_{max}]\)$
- $\(D \in [D_{min}, D_{max}]\)$
- $\(\mu \in [\mu_{min}, \mu_{max}]\)$

### Initial Condition Families

Three representative families are considered:

- **Tanh**
- **Gaussian**
- **Sin-Gauss**

These families were chosen to cover smooth fronts, localized pulses, and oscillatory localized structures.

### Domain Discretization

The computational domain is defined by:

- spatial domain: $\(x \in [x_{min}, x_{max}]\)$
- temporal horizon: $\(t \in [0, T_{max}]\)$

with dedicated numerical grids:

- spatial resolution: $\(N_x\)$
- temporal resolution: $\(N_t\)$

The objective is to evaluate whether a single surrogate can generalize across both physical parameters and initial-condition structures.

---

## Step 2 — Classical Reference Solver

A robust classical numerical solver is implemented as the scientific ground truth.

### Reference Method

We use a Crank-Nicolson finite-difference scheme for the ADR equation.

### Validation Procedure

The reference solver is validated through:

- mesh refinement studies  
- temporal refinement studies  
- convergence checks  
- numerical stability checks

Whenever possible, solutions are also compared against standard references from classical PDE literature.

### Role in This Repository

The Crank-Nicolson solver provides:

- training supervision when required  
- final benchmark targets  
- speedup comparisons against neural surrogates

---

## Step 3 — Single-Case PI-DeepONet Validation

Before attempting generalization, the PI-DeepONet is tested on simplified single-case settings.

### Protocol

A single fixed triplet:

- one value of $\(v\)$
- one value of $\(D\)$
- one value of $\(\mu\)$

is selected.

Three independent experiments are run:

- one Tanh initial condition  
- one Gaussian initial condition  
- one Sin-Gauss initial condition

### Objective

Validate that the architecture can solve the ADR equation in controlled conditions before scaling to broader distributions.

---

## Step 4 — Multi-Family Generalization (Single Physics Case)

The model is then extended to handle multiple initial-condition families simultaneously while keeping physics coefficients fixed.

### Objective

Train one PI-DeepONet capable of solving:

- Tanh cases  
- Gaussian cases  
- Sin-Gauss cases

under the same fixed $\((v,D,\mu)\)$.

This tests representational robustness across heterogeneous initial conditions.

---

## Step 5 — Full Parametric Generalization

The surrogate is then trained on varying coefficients and varying initial conditions jointly.

### Training Distribution

Samples are generated across:

- varying $\(v\)$
- varying $\(D\)$
- varying $\(\mu\)$
- all three initial-condition families

### Objective

Learn a parametric solution operator:

$$
(v, D, \mu, u_0(x)) \rightarrow u(x,t)
$$

rather than a single PDE instance.

---

## Step 6 — Hyperparameter Optimization with Optuna

After validating the pipeline, Optuna is used to optimize the training setup.

### Search Targets

The study balances:

- final relative $\(L^2\)$ error  
- training stability  
- wall-clock training time  
- inference speed

### Search Space Includes

Examples:

- learning rate  
- depth / width  
- Fourier features  
- loss weights  
- scheduler settings  
- batch size

---

## Step 7 — Final Frozen Run

The best Optuna configuration is frozen.

A final production run is launched using these selected hyperparameters.

### Objective

Produce the canonical benchmark result reported in the repository.

This run serves as the main scientific conclusion for the PI-DeepONet baseline.

---

# Stage B — PyTorch vs JAX Benchmark

---

## Step 8 — Reproducing the Pipeline in JAX

The validated PyTorch training logic is reimplemented in JAX.

### Matching Principles

The comparison preserves:

- same architecture  
- same parameter ranges  
- same datasets  
- same loss structure  
- same training curriculum  
- same evaluation metrics

### Objective

Measure whether JAX provides better speed/accuracy tradeoffs under matched conditions.

---

## Step 9 — Monofamily Benchmarks

Both frameworks are evaluated on simplified monofamily settings:

- Tanh only  
- Gaussian only  
- Sin-Gauss only

### Objective

Determine whether some framework differences are hidden by the harder multifamily benchmark.

---

## Step 10 — Ansatz Ablation Study

Monofamily experiments are repeated with and without handcrafted ansatz priors.

### Objective

Quantify the effect of inductive bias versus free learning.

This identifies whether architecture priors matter more than backend choice.

---

## Step 11 — L-BFGS Finisher Study

Both frameworks are evaluated with and without a final L-BFGS optimization stage.

### Objective

Measure whether second-stage optimization materially improves convergence.

---

# Final Evaluation Metrics

All experiments are assessed with:

- global relative $\(L^2\)$ error  
- family-wise relative $\(L^2\)$ error  
- training time  
- inference time  
- speedup vs Crank-Nicolson  
- convergence robustness across seeds

---

# Final Research Questions

## Scientific Question (Steps 1–7)

Can a PI-DeepONet become a robust and useful surrogate solver for this ADR problem?

## Engineering Question (Steps 8–11)

For this specific scientific ML task, which framework provides the best tradeoff between:

- final accuracy  
- robustness  
- training speed  
- deployment practicality

between **PyTorch** and **JAX**?
