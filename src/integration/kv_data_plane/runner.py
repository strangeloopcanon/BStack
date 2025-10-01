from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from bw_apis import CachePlan, KvPageRef, TransferKind, TransferOp, cache_plan
from bw_stack.paths import add_third_party_to_path, resolve

add_third_party_to_path()

from bodocache.config import RuntimeConfig, load_config_typed
from bodocache.planner.cluster import assign_pclusters_minhash
from bodocache.planner.scheduler import run_window
from bodocache.sim.utils import (
    synthetic_heat,
    synthetic_layer_lat,
    synthetic_requests,
    synthetic_tenant_caps,
    synthetic_tier_caps,
)
from bodocache.agent.sim_node import simulate_plan_streams, summarize_metrics


@dataclass
class CachePlanResult:
    plan: CachePlan
    plan_df: pd.DataFrame
    evict_df: pd.DataFrame
    admission_df: pd.DataFrame
    tiers_df: pd.DataFrame
    layer_lat_df: pd.DataFrame
    cfg: RuntimeConfig


def build_cache_plan(*, now_ms: Optional[int] = None, window_id: Optional[str] = None, request_count: int = 200) -> CachePlanResult:
    """Generate a CachePlan using the synthetic BCache workload."""

    os.environ.setdefault("BODOCACHE_PURE_PY", "1")

    cfg = load_config_typed(runtime_path=str(resolve("third_party", "BCache", "configs", "runtime.yaml")))
    req = synthetic_requests(n_req=request_count)

    if cfg.ab_flags.enable_prefix_fanout:
        req = assign_pclusters_minhash(req, num_hashes=32, bands=8, k=4)
    else:
        req = req.copy()
        req["pcluster"] = req["req_id"].astype(int)

    heat = synthetic_heat(req)
    tiers = synthetic_tier_caps()
    lats = synthetic_layer_lat()
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    tenant_caps = synthetic_tenant_caps(req["tenant"], cfg.tenant_credits_bytes)

    plan_df, evict_df, admission_df = run_window(
        req,
        heat,
        tiers,
        tenant_caps,
        lats,
        now_ms=now_ms,
        pmin=cfg.thresholds.pmin,
        umin=cfg.thresholds.umin,
        min_io_bytes=cfg.min_io_bytes,
        alpha=cfg.popularity.alpha,
        beta=cfg.popularity.beta,
        window_ms=cfg.window_ms,
        max_ops_per_tier=cfg.max_ops_per_tier,
        enable_admission=cfg.ab_flags.enable_admission,
        enable_eviction=cfg.ab_flags.enable_eviction,
        enforce_tier_caps=cfg.ab_flags.enforce_tier_caps,
    )

    plan_id = window_id or f"cache-{now_ms}"
    api_plan = _convert_to_cache_plan(plan_id, plan_df, evict_df, admission_df)
    return CachePlanResult(
        plan=api_plan,
        plan_df=plan_df,
        evict_df=evict_df,
        admission_df=admission_df,
        tiers_df=tiers,
        layer_lat_df=lats,
        cfg=cfg,
    )


def simulate_cache_plan(result: CachePlanResult) -> Dict[str, float]:
    """Feed the plan to the built-in multistream simulator to obtain metrics."""

    exec_df = simulate_plan_streams(
        result.plan_df,
        result.tiers_df,
        window_ms=int(result.cfg.window_ms),
        streams_per_tier=4,
        use_overlap=result.cfg.ab_flags.enable_overlap,
        layer_lat_df=result.layer_lat_df,
    )
    summary = summarize_metrics(exec_df)
    return {
        "ops": float(summary["ops"]),
        "avg_finish_ms": float(summary["avg_finish_ms"]),
        "prefetch_timeliness": float(summary["prefetch_timeliness"]),
    }


def _convert_to_cache_plan(plan_id: str, plan_df: pd.DataFrame, evict_df: pd.DataFrame, admission_df: pd.DataFrame) -> CachePlan:
    ops = []
    for row in plan_df.itertuples(index=False):
        tier_src = int(getattr(row, "tier_src", 0))
        tier_dst = int(getattr(row, "tier_dst", 0))
        kind = _infer_kind(tier_src, tier_dst)
        src = f"tier://{getattr(row, 'node', 'node-0')}/tier{tier_src}"
        dst = f"tier://{getattr(row, 'node', 'node-0')}/tier{tier_dst}"
        start_pid = int(getattr(row, "start_pid", 0))
        end_pid = int(getattr(row, "end_pid", start_pid))
        page_bytes = int(getattr(row, "page_bytes", 256 * 1024))
        kv_refs = [
            KvPageRef(tensor="kv", page=pid, head=0, layer=int(getattr(row, "layer", 0)))
            for pid in range(start_pid, end_pid + 1)
        ]
        note = f"cluster={getattr(row, 'pcluster', 0)} fanout={getattr(row, 'fanout', 1)} overlap={getattr(row, 'overlap', 1)}"
        ops.append(
            TransferOp(
                kind=kind,
                src=src,
                dst=dst,
                length=int(getattr(row, "bytes", 0)),
                src_offset=start_pid * page_bytes,
                dst_offset=start_pid * page_bytes,
                kv_refs=kv_refs,
                note=note,
            )
        )

    prefetch = [
        KvPageRef(tensor="kv", page=int(getattr(row, "page_id", 0)), head=0, layer=int(getattr(row, "layer", 0)))
        for row in admission_df.itertuples(index=False)
    ]
    evict = [
        KvPageRef(tensor="kv", page=int(getattr(row, "page_id", 0)), head=0, layer=int(getattr(row, "layer", 0)))
        for row in evict_df.itertuples(index=False)
    ]

    return cache_plan(plan_id, ops, prefetch=prefetch, evict=evict)


def _infer_kind(tier_src: int, tier_dst: int) -> TransferKind:
    if tier_src < tier_dst:
        return TransferKind.H2D
    if tier_src > tier_dst:
        return TransferKind.D2H
    return TransferKind.P2P
