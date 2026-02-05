from __future__ import annotations

import importlib

import pytest


def test_integration_modules_importable() -> None:
    for name in (
        "integration.data_pipeline.datajax_bridge",
        "integration.kv_data_plane.runner",
        "integration.weight_swapper.runner",
        "integration.examples.run_stack",
    ):
        importlib.import_module(name)


def test_run_stack_help_smoke() -> None:
    run_stack = importlib.import_module("integration.examples.run_stack")
    with pytest.raises(SystemExit) as exc:
        run_stack.main(["--help"])
    assert exc.value.code == 0
