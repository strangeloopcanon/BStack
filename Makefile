PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: submodules dev-latest bootstrap codegen sync test lint fmt examples clean

submodules:
	git submodule update --init --recursive

dev-latest:
	git submodule foreach 'git fetch --tags origin'
	git submodule update --remote --merge
	@echo "Updated submodules to latest remote HEADs (not committed)."

bootstrap: submodules
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip setuptools wheel
	$(BIN)/pip install "numpy>=1.24,<2.3"
	$(BIN)/pip install -e .[dev]
	$(BIN)/pip install -e third_party/hotweights
	$(BIN)/pip install -e third_party/BCache
	$(BIN)/pip install -e third_party/datajax
	$(BIN)/pip install -e third_party/bw-runtime/python

codegen:
	$(BIN)/python scripts/codegen.py

sync:
	@$(BIN)/python scripts/check_lock.py

lint:
	$(BIN)/ruff check src/bstack src/bstack_apis src/integration tests

fmt:
	$(BIN)/ruff format src/bstack src/bstack_apis src/integration tests

pytest:
	$(BIN)/pytest -m "not gpu" tests

test: pytest

examples:
	$(BIN)/python -m integration.examples.run_stack

clean:
	rm -rf $(VENV) build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name "__pycache__" -exec rm -r {} +
