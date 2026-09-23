"""KV-cache hierarchical tiering demo.

Generates a synthetic LLM-serving trace (shared system prefixes + Zipf
context lengths), runs BCache's planner window-by-window against a simulated
STORAGE -> CPU -> GPU hierarchy, and compares it with a naive FIFO-prefetch
baseline. Prints the verdict and writes machine-readable results to out/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bstack.paths import add_third_party_to_path, resolve

add_third_party_to_path()

from bstack_apis import (
    KvPageRef,
    TransferKind,
    TransferOp,
    cache_plan,
)
from integration.kv_tiering import (
    TieringConfig,
    TraceConfig,
    generate_trace,
    run_tiering_experiment,
    trace_stats,
)
from integration.kv_tiering.tiering import (
    TIER_NAMES,
    summarize_comparison,
    ungid,
)


def _to_cache_plan(result, plan_id: str):
    ops = []
    for op in result.last_ops:
        kind = TransferKind.H2D if op.tier_src < op.tier_dst else TransferKind.D2H
        kv_refs = [
            KvPageRef(tensor="kv", page=pid, head=0, layer=op.layer)
            for pid in range(op.start_pid, op.end_pid + 1)
        ]
        ops.append(
            TransferOp(
                kind=kind,
                src=f"tier://node-0/tier{op.tier_src}",
                dst=f"tier://node-0/tier{op.tier_dst}",
                length=op.nbytes,
                src_offset=op.start_pid * op.page_bytes,
                dst_offset=op.start_pid * op.page_bytes,
                kv_refs=kv_refs,
                note=f"session={op.pcluster} layer={op.layer}",
            )
        )
    return cache_plan(plan_id, ops)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the KV-cache tiering demo")
    parser.add_argument("--output", type=Path, default=resolve("out"))
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--requests", type=int, default=600)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument(
        "--gpu-pages", type=int, default=4096, help="GPU tier capacity in page-slots"
    )
    parser.add_argument(
        "--bandwidth-mb",
        type=int,
        default=256,
        help="Prefetch bandwidth budget per window, in MB",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    tcfg = TraceConfig(
        n_sessions=args.sessions,
        n_requests=args.requests,
        n_layers=args.layers,
        n_windows=args.windows,
        seed=args.seed,
    )
    tier_cfg = TieringConfig(
        gpu_capacity_pages=args.gpu_pages,
        window_bandwidth_bytes=args.bandwidth_mb * 1024 * 1024,
    )

    print("Generating workload trace ...")
    trace = generate_trace(tcfg)
    stats = trace_stats(trace)
    print(
        f"  {stats['n_requests']} requests, {stats['n_sessions']} sessions, "
        f"mean prefix reuse per page: {stats['mean_reuse_per_page']}x"
    )

    print("Running BCache planner policy ...")
    planner_res = run_tiering_experiment(trace, tier_cfg, policy="planner")
    print("Running naive FIFO baseline ...")
    naive_res = run_tiering_experiment(trace, tier_cfg, policy="naive")

    comp = summarize_comparison(planner_res, naive_res)
    print()
    print(
        f"{'policy':<10}{'hit':>8}{'prefix':>8}{'interact':>10}{'served':>8}{'GB':>8}{'ops':>6}{'ms':>8}"
    )
    for name in ("planner", "naive"):
        c = comp[name]
        print(
            f"{name:<10}{c['mean_hit_rate']:>8.3f}{c['mean_prefix_hit_rate']:>8.3f}"
            f"{c['mean_interactive_hit_rate']:>10.3f}{c['mean_interactive_served_rate']:>8.3f}"
            f"{c['total_planned_gb']:>8.2f}{c['mean_ops']:>6.1f}{c['mean_planner_ms']:>8.1f}"
        )
    lift = (
        comp["planner"]["mean_interactive_served_rate"]
        - comp["naive"]["mean_interactive_served_rate"]
    )
    print(
        f"\nPlanner interactive-served lift over naive: {lift:+.3f} "
        f"(deadline-ordered prefetch vs arrival-order demand paging)"
    )

    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "trace": stats,
        "config": {
            "gpu_capacity_pages": tier_cfg.gpu_capacity_pages,
            "window_bandwidth_mb": args.bandwidth_mb,
            "seed": args.seed,
        },
        "comparison": comp,
        "interactive_served_lift": round(lift, 4),
        "planner_windows": [w.__dict__ for w in planner_res.windows],
        "naive_windows": [w.__dict__ for w in naive_res.windows],
    }
    (out_dir / "kv_tiering.json").write_text(json.dumps(report, indent=2))
    plan = _to_cache_plan(planner_res, "kv-tiering-final-window")
    plan.to_json(out_dir / "kv_tiering_plan.json")
    print(
        f"\nWrote {out_dir / 'kv_tiering.json'} and {out_dir / 'kv_tiering_plan.json'}"
    )
    print(f"Tier map: {TIER_NAMES} | ungid check: {ungid(3 * 4096 + 11)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
