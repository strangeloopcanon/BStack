PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: submodules dev-latest bootstrap codegen sync test lint fmt examples clean bench demo-kv demo-swap demos

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
	$(BIN)/pip install -e third_party/bw-runtime

codegen:
	$(BIN)/python scripts/codegen.py

sync:
	@$(BIN)/python scripts/check_lock.py

lint:
	$(BIN)/ruff check src/bstack src/bstack_apis src/integration tests

fmt:
	$(BIN)/ruff format src/bstack src/bstack_apis src/integration tests

pytest:
	$(BIN)/pytest -m "not gpu" tests src/integration/bench

test: pytest

bench:
	$(BIN)/pytest -m "bench" src/integration/bench -s

examples:
	$(BIN)/python -m integration.examples.run_stack

demo-kv:
	$(BIN)/python -m integration.examples.run_kv_tiering

demo-swap:
	$(BIN)/python -m integration.examples.run_swap_kv

demos: examples demo-kv demo-swap

clean:
	rm -rf $(VENV) build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name "__pycache__" -exec rm -r {} +
