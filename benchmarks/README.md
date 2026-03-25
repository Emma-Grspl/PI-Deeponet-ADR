# Benchmark Layout

This benchmark stack is isolated from the baseline training scripts.

Goals:
- compare PyTorch and JAX under the same short-horizon protocol
- measure training speed, convergence, CN-relative L2, inference speed, and time-jump speed
- keep outputs under `outputs/benchmarks/`

Suggested workflow:
1. run `benchmarks/pytorch/train_short_benchmark.py`
2. run `benchmarks/jax/train_short_benchmark.py`
3. run the corresponding `eval_benchmark.py`
4. run the corresponding `inference_benchmark.py`
5. aggregate with `benchmarks/aggregate_results.py`
