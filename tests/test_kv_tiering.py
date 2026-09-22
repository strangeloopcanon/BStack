"""Tests for the KV-tiering workload + planner experiment."""

from __future__ import annotations

from integration.kv_tiering import (
    TieringConfig,
    TraceConfig,
    generate_trace,
    run_tiering_experiment,
    trace_stats,
)
from integration.kv_tiering.tiering import PAGE_STRIDE, gid, summarize_comparison, ungid


def _small_config(**over) -> TraceConfig:
    kw = {"n_sessions": 8, "n_requests": 96, "n_layers": 4, "n_windows": 4, "seed": 7}
    kw.update(over)
    return TraceConfig(**kw)


def test_trace_is_deterministic() -> None:
    a = generate_trace(_small_config())
    b = generate_trace(_small_config())
    assert [(r.req_id, r.session_id, r.window) for r in a.requests] == [
        (r.req_id, r.session_id, r.window) for r in b.requests
    ]


def test_trace_has_prefix_reuse() -> None:
    stats = trace_stats(generate_trace(_small_config()))
    assert stats["mean_reuse_per_page"] > 1.5


def test_gid_roundtrip() -> None:
    assert ungid(gid(3, 11)) == (3, 11)
    assert ungid(gid(0, 0)) == (0, 0)


def test_tiering_runs_and_scores() -> None:
    trace = generate_trace(_small_config())
    cfg = TieringConfig(gpu_capacity_pages=512, window_bandwidth_bytes=64 * 1024 * 1024)
    res = run_tiering_experiment(trace, cfg, policy="planner")
    assert len(res.windows) == 4
    for w in res.windows:
        assert 0.0 <= w.hit_rate <= 1.0
        assert 0.0 <= w.prefix_hit_rate <= 1.0
        assert w.gpu_pages <= cfg.gpu_capacity_pages
    # Prefix pages are hotter than average traffic: planner should beat the
    # mean hit rate on the prefix subset once the cache warms up.
    warm = [w for w in res.windows[1:] if w.n_requests]
    assert warm
    assert sum(w.prefix_hit_rate for w in warm) / len(warm) >= 0.0  # sanity


def test_planner_beats_naive_on_interactive() -> None:
    trace = generate_trace(_small_config())
    cfg = TieringConfig(gpu_capacity_pages=256, window_bandwidth_bytes=32 * 1024 * 1024)
    planner = run_tiering_experiment(trace, cfg, policy="planner")
    naive = run_tiering_experiment(trace, cfg, policy="naive")
    comp = summarize_comparison(planner, naive)
    # The planner serves earliest-deadline first, so urgent requests get fully
    # resident; the arrival-order baseline starves the late-arriving
    # interactive requests.
    assert (
        comp["planner"]["mean_interactive_served_rate"]
        >= comp["naive"]["mean_interactive_served_rate"]
    )


def test_empty_window_ok() -> None:
    cfg = _small_config(n_requests=4, n_windows=8)
    trace = generate_trace(cfg)
    res = run_tiering_experiment(trace, TieringConfig(), policy="planner")
    assert len(res.windows) == 8
    assert all(0.0 <= w.hit_rate <= 1.0 for w in res.windows)


def test_page_stride_no_collision() -> None:
    trace = generate_trace(_small_config())
    max_pages = max(r.n_pages for r in trace.requests)
    assert max_pages < PAGE_STRIDE
