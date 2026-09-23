# BStack Integration Repo

Umbrella integration for the Strange Loop Canon serving stack:

- [hotweights](https://github.com/strangeloopcanon/hotweights) — versioned model-weight updates
- [BCache](https://github.com/strangeloopcanon/BCache) — hierarchical KV-cache planner
- [datajax](https://github.com/strangeloopcanon/datajax) — data/plan execution
- [bw-runtime](https://github.com/strangeloopcanon/bw-runtime) — C++ runtime

BStack keeps each component in its upstream repo (pinned via submodules + `stack.lock`)
and adds the thin layer that makes them work together: a shared plan IR
(`src/bstack_apis/`), integration glue, and end-to-end demos that exercise the
seams — especially the KV-cache path.

## Demos

### 1. KV-cache hierarchical tiering (`make demo-kv`)

A seeded synthetic LLM serving trace (configurable sessions, Zipf-distributed
context lengths, shared system/session prefixes, mixed interactive + batch
requests with deadlines) is fed window-by-window into the **real BCache
`plan_window` planner**. A simulated 3-tier hierarchy (storage → CPU → GPU/HBM)
applies the plans, tracks residency/hit-rate/bytes-moved, and compares against
an arrival-order demand-paging baseline.

The headline result: under a fixed per-window transfer budget, the planner's
deadline-ordered prefetch fully serves urgent requests that the baseline
starves.

```
policy         hit  prefix  interact  served      GB   ops      ms
planner      0.230   0.273     0.909   0.889    2.30  23.8   163.0
naive        0.303   0.375     0.000   0.000    2.36  33.9     0.2
```

Measured with the default config (600 requests, 40 sessions, 8 layers,
12 windows, seed 7): planner interactive-served lift over naive **+0.889**.
`served` = fraction of interactive requests ≥95% GPU-resident after the plan.
The naive baseline moves a similar number of bytes but spends them on
batch demand in arrival order, so every urgent request stalls.

Outputs: `out/kv_tiering.json` (per-window metrics + comparison),
`out/kv_tiering_plan.json` (final window's plan in the shared IR).

### 2. Weight-swap × KV-cache reuse (`make demo-swap`)

Builds two synthetic hotweights checkpoints, diffs them with the real
hotweights manifest/`create_plan` APIs, maps changed tensors to affected
transformer layers, and emits **selective KV invalidation** using the shared
`CachePlan.evict` IR — instead of flushing the whole KV cache on a weight
update.

Default (8 layers, layers 2 and 5 changed):

```
changed layers:      [2, 5]
KV cache total:      134.22 MB
KV invalidated:      33.55 MB
KV reusable:         100.66 MB (75%)
```

Outputs: `out/swap_kv.json`, `out/swap_kv_plan.json`.

### 3. Original stack demo (`make examples`)

The earlier synthetic end-to-end demo (BCache cache plans + hotweights swap
plans + DataJAX analytics + optional bw-runtime probe). Kept for regression;
the two demos above are the interesting ones now.

## Quick start

```bash
git submodule update --init --recursive   # or: git clone --recurse-submodules
make bootstrap   # venv + editable installs (also inits submodules)
make test        # 19 tests, CPU-only
make demos       # demo-kv + demo-swap + examples
make bench       # planner benchmarks (CPU-only, <2 min)
make lint        # ruff check
```

## Benchmarks (`make bench`)

CPU-only planner benchmarks; three workload sizes, all under two minutes.
Writes `out/bench_planner.json`.

| case   | requests | total time | throughput | mean plan/window |
|--------|----------|-----------|------------|------------------|
| small  | 200      | 0.62 s    | 323 req/s  | 92.7 ms          |
| medium | 1,000    | 0.83 s    | 1,203 req/s| 94.7 ms          |
| large  | 3,000    | 1.49 s    | 2,017 req/s| 133.9 ms         |

(Measured 2026-09-22 on the dev VM; rerun `make bench` for your numbers.)

## Layout

- `src/bstack_apis/` — shared plan IR (protobuf schema, Python + C++ helpers).
- `src/integration/kv_tiering/` — workload generator + tiering simulator
  (`workload.py`, `tiering.py`).
- `src/integration/swap_analysis/` — weight-delta → KV invalidation
  (`swap_kv.py`).
- `src/integration/data_pipeline/datajax_bridge.py` — DataJAX plan
  introspection (`describe` + `explain`).
- `src/integration/examples/` — demo runners (`run_kv_tiering.py`,
  `run_swap_kv.py`, `run_stack.py`).
- `src/integration/bench/` — pytest benchmarks (`test_planner_bench.py`).
- `tests/` — integration tests (`test_kv_tiering.py`, `test_swap_kv.py`,
  `test_bstack_apis.py`).
- `third_party/` — pinned submodules; `stack.lock` records SHAs,
  `make sync` verifies them.

## Submodule policy

Submodules live in `third_party/` and are locked to explicit SHAs. Modify a
component upstream, then bump the SHA plus `stack.lock`; never leave
long-lived patches inside `third_party/`.

## Runtime notes

- The integration forces pure-Python fallback paths (`BODOCACHE_PURE_PY=1`,
  `HOTWEIGHTS_FORCE_PANDAS=1`) so the demos run without Bodo.
- Bootstrap pins `numpy<2.3` to avoid SciPy/Numba compatibility warnings.
- `bw-runtime` requires a compiled shared library; the demo probes the Python
  bindings and reports if it is missing.
- The tiering simulator models transfer budgets and capacity, not interconnect
  latency or kernel launch overhead; treat the numbers as planner-behavior
  comparisons, not hardware predictions.

## License

Apache-2.0 — see `LICENSE`.
