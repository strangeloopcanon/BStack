"""Planner benchmarks: BCache plan_window latency/throughput vs trace size.

CPU-only. The whole module must finish in well under 2 minutes; each case
asserts a generous per-case ceiling so regressions are caught, not just
measured. Run with ``pytest tests -m bench -s`` to see the table, or
``make bench`` to also persist JSON.
"""

from __future__ import annotations

import json
import time

import pytest

from bstack.paths import add_third_party_to_path, resolve

add_third_party_to_path()

from integration.kv_tiering import TraceConfig, generate_trace
from integration.kv_tiering.tiering import (
    TieringConfig,
    run_tiering_experiment,
)

pytestmark = pytest.mark.bench

CASES = [
    # (label, n_requests, n_sessions, ceiling_seconds)
    ("small", 200, 16, 60),
    ("medium", 1000, 40, 90),
    ("large", 3000, 80, 120),
]

RESULTS: list[dict] = []


@pytest.mark.parametrize("label,n_requests,n_sessions,ceiling", CASES)
def test_planner_latency(
    label: str, n_requests: int, n_sessions: int, ceiling: float
) -> None:
    tcfg = TraceConfig(
        n_sessions=n_sessions,
        n_requests=n_requests,
        n_layers=8,
        n_windows=6,
        seed=7,
    )
    tier_cfg = TieringConfig(
        gpu_capacity_pages=2048, window_bandwidth_bytes=128 * 1024 * 1024
    )
    trace = generate_trace(tcfg)

    t0 = time.perf_counter()
    res = run_tiering_experiment(trace, tier_cfg, policy="planner")
    elapsed = time.perf_counter() - t0

    plan_ms = [w.planner_ms for w in res.windows]
    row = {
        "case": label,
        "n_requests": n_requests,
        "elapsed_s": round(elapsed, 2),
        "req_per_s": round(n_requests / elapsed, 1),
        "mean_plan_ms": round(sum(plan_ms) / len(plan_ms), 1),
        "max_plan_ms": round(max(plan_ms), 1),
        "mean_prefix_hit": round(res.mean_prefix_hit_rate, 3),
    }
    RESULTS.append(row)
    print(
        f"\n[bench:{label}] {row['elapsed_s']}s total, "
        f"{row['req_per_s']} req/s, mean plan {row['mean_plan_ms']}ms, "
        f"prefix hit {row['mean_prefix_hit']}"
    )
    assert elapsed < ceiling, f"{label} took {elapsed:.1f}s > {ceiling}s ceiling"


def test_bench_write_report() -> None:
    assert RESULTS, "no benchmark cases ran"
    out = resolve("out")
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench_planner.json").write_text(json.dumps(RESULTS, indent=2))
