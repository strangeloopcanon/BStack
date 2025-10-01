# BStack Umbrella

This repository composes the `hotweights`, `BCache`, `datajax`, and `bstack-runtime` projects behind a thin shared API. It locks component SHAs via submodules and publishes integration demos so you can inspect the seams without collapsing into a monorepo.

## Quick Start

```bash
git submodule update --init --recursive  # fetch third_party repos (or use: git clone --recurse-submodules)
make bootstrap        # create .venv and install all editable packages (also initializes submodules)
make codegen          # regenerate protobuf bindings (requires grpcio-tools)
make test             # run bstack_apis unit tests
make examples         # produce CachePlan + SwapPlan JSON under out/
```

The demo emits two files:

- `out/cache_plan.json`: synthetic cache transfer plan produced by BCache and normalized to the shared IR.
- `out/swap_plan.json`: weight swap plan produced by hotweights from two demo checkpoints.

## Layout

- `src/bstack_apis/` — shared plan IR (protobuf schema, Python + C++ helpers).
- `src/integration/` — integration code for submodules.
- `src/integration/examples/run_stack.py` — orchestrates the end-to-end demo.
- `ops/` — placeholders for future Docker/compose/Grafana assets.
- `stack.lock` — pins submodule SHAs; `make sync` verifies they match.

## Submodule Policy

Submodules live in `third_party/` and are locked to explicit SHAs. Modify a component upstream and then bump the SHA plus `stack.lock`; never leave long-lived patches inside `third_party/`.

## Runtime Notes

- The integration forces pure-Python fallback paths (`BODOCACHE_PURE_PY=1`, `HOTWEIGHTS_FORCE_PANDAS=1`) so the demo runs without Bodo.
- `bstack-runtime` requires a compiled shared library. The demo script probes the Python bindings and reports if the library is missing.
- `datajax` is installed for completeness; glue code will live in future iterations once plan consumers iterate on the IR.

## Next Steps

1. Flesh out protobuf code generation (publish wheel/tarball with generated stubs).
2. Replace synthetic workloads with actual datajax traces feeding BCache planners.
3. Wire `bstack-runtime` execution once PCIe/GPU paths are available or a CPU fallback lands upstream.
4. Extend `integration/bench/` with reproducible benchmarks gated in CI.