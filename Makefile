PYTHON ?= python

.PHONY: install test check analysis benchmark train

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-base.txt pytest
	$(PYTHON) -m pip install -r requirements-jax.txt

test:
	$(PYTHON) -m pytest -q base/code/tests

check:
	$(PYTHON) -m compileall base/code/scripts base/code/src base/code/tests jax_vs_pytorch/code/benchmarks jax_vs_pytorch/code/code_experiments jax_vs_pytorch/code/src_jax

analysis:
	$(PYTHON) base/code/src/analyse/global_analyse_PI_DeepOnet_vs_CN.py

benchmark:
	$(PYTHON) jax_vs_pytorch/code/benchmarks/aggregate_results.py

train:
	$(PYTHON) base/code/scripts/train.py
