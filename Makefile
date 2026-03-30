PYTHON ?= python

.PHONY: install test check analysis benchmark train

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt pytest

test:
	$(PYTHON) -m pytest -q

check:
	$(PYTHON) -m compileall scripts src test

analysis:
	$(PYTHON) src/analyse/global_analyse_PI_DeepOnet_vs_CN.py

benchmark:
	$(PYTHON) src/analyse/inference.py

train:
	$(PYTHON) scripts/train.py
