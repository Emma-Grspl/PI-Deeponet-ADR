PYTHON ?= python

.PHONY: install test check analysis benchmark train

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-base.txt pytest

test:
	$(PYTHON) -m pytest -q code/tests

check:
	$(PYTHON) -m compileall code/scripts code/src code/tests

analysis:
	$(PYTHON) code/src/analyse/global_analyse_PI_DeepOnet_vs_CN.py

benchmark:
	$(PYTHON) code/src/analyse/inference.py

train:
	$(PYTHON) code/scripts/train.py
