"""Hierarchical KV-cache tiering experiment.

Feeds a realistic workload trace into BCache's planner window-by-window,
maintains a page-table simulation of a 3-tier hierarchy
(STORAGE=0 / CPU=1 / GPU=2), and scores how well the planner keeps hot
prefix pages in GPU HBM under bandwidth and capacity pressure.

A naive FIFO-prefetch baseline runs through the same simulator so the
planner's heat-aware decisions can be compared apples-to-apples.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

os.environ.setdefault("BODOCACHE_PURE_PY", "1")

from bstack.paths import add_third_party_to_path

add_third_party_to_path()

from bodocache.planner.api import (
    HeatEntry,
    PlannerConfig,
    PlannerRequest,
    PlannerResult,
    PlannerWindow,
    TierCapacity,
    plan_window,
)

from integration.kv_tiering.workload import WorkloadTrace

# Stride separating sessions in the planner's flat page-id space, so an
# eviction of (layer, page_id) maps back to exactly one (session, page).
PAGE_STRIDE = 4096

TIER_NAMES = {0: "STORAGE", 1: "CPU", 2: "GPU"}


def gid(session_id: int, local_page: int) -> int:
    return session_id * PAGE_STRIDE + local_page


def ungid(flat: int) -> tuple[int, int]:
    return divmod(flat, PAGE_STRIDE)


@dataclass
class TieringConfig:
    gpu_capacity_pages: int = 4096  # page-slots in tier 2 (shared across layers)
    window_bandwidth_bytes: int = 256 * 1024 * 1024  # prefetch budget per window
    max_ops_per_tier: int = 128
    min_io_bytes: int = 512 * 1024
    # Background eviction (the node agent's job): keep GPU under this fill
    # fraction by evicting coldest pages before planning each window.
    evict_watermark: float = 0.85
    # pcluster="session" lets the planner coalesce a session's shared prefix
    # into fan-out runs; "request" disables it (ablation).
    pcluster_mode: str = "session"


@dataclass
class WindowMetrics:
    window: int
    n_requests: int
    hit_rate: float  # fraction of needed pages already in GPU (pre-plan)
    prefix_hit_rate: float  # same, restricted to shared prefix pages
    interactive_hit_rate: float  # interactive requests' pages resident AFTER the plan
    interactive_served_rate: float  # fraction of interactive requests >=95% resident
    planned_bytes: int
    evicted_bytes: int
    n_ops: int
    gpu_pages: int
    planner_ms: float


@dataclass
class TieringResult:
    policy: str
    windows: list[WindowMetrics] = field(default_factory=list)
    total_planned_bytes: int = 0
    total_evicted_bytes: int = 0
    last_ops: list[_Op] = field(default_factory=list)  # ops of the final window

    @property
    def mean_hit_rate(self) -> float:
        vals = [w.hit_rate for w in self.windows if w.n_requests]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_prefix_hit_rate(self) -> float:
        vals = [w.prefix_hit_rate for w in self.windows if w.n_requests]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_interactive_hit_rate(self) -> float:
        vals = [w.interactive_hit_rate for w in self.windows if w.n_requests]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def mean_interactive_served_rate(self) -> float:
        vals = [w.interactive_served_rate for w in self.windows if w.n_requests]
        return sum(vals) / len(vals) if vals else 0.0


@dataclass
class _Op:
    pcluster: int  # session_id
    layer: int
    tier_src: int
    tier_dst: int
    start_pid: int
    end_pid: int
    page_bytes: int

    @property
    def nbytes(self) -> int:
        return (self.end_pid - self.start_pid + 1) * self.page_bytes


class _PageTable:
    """Ground-truth residency: (session, layer, flat_page) -> tier."""

    def __init__(self) -> None:
        self.tiers: dict[tuple[int, int, int], int] = {}
        self.hits: dict[tuple[int, int, int], int] = {}

    def get(self, key: tuple[int, int, int]) -> int:
        return self.tiers.get(key, 0)

    def gpu_pages(self) -> int:
        return sum(1 for t in self.tiers.values() if t == 2)


def _coalesce(sorted_ids: list[int]) -> list[tuple[int, int]]:
    """Merge sorted page ids into [start, end] intervals."""
    if not sorted_ids:
        return []
    out: list[tuple[int, int]] = []
    start = prev = sorted_ids[0]
    for pid in sorted_ids[1:]:
        if pid == prev + 1:
            prev = pid
        else:
            out.append((start, prev))
            start = prev = pid
    out.append((start, prev))
    return out


def _planner_policy(
    intervals: list[dict],
    heat: list[HeatEntry],
    tier_caps: list[TierCapacity],
    now_ms: int,
    cfg: TieringConfig,
) -> tuple[list[_Op], list[tuple[int, int]], float]:
    """Run BCache's planner; return (ops, evictions as (layer, flat_page), ms)."""
    window = PlannerWindow(
        requests=[PlannerRequest(**kw) for kw in intervals],
        now_ms=now_ms,
        heat=heat,
        tier_caps=tier_caps,
    )
    t0 = time.perf_counter()
    result: PlannerResult = plan_window(
        window,
        PlannerConfig(
            max_ops_per_tier=cfg.max_ops_per_tier,
            min_io_bytes=cfg.min_io_bytes,
        ),
    )
    ms = (time.perf_counter() - t0) * 1000.0
    ops = [
        _Op(
            pcluster=op.pcluster,
            layer=op.layer,
            tier_src=op.tier_src,
            tier_dst=op.tier_dst,
            start_pid=op.start_pid or 0,
            end_pid=op.end_pid or 0,
            page_bytes=op.page_bytes or 0,
        )
        for op in result.plan
    ]
    evictions = [(e.layer, e.page_id) for e in result.evictions]
    return ops, evictions, ms


def _naive_policy(
    intervals: list[dict],
    table: _PageTable,
    cfg: TieringConfig,
) -> tuple[list[_Op], list[tuple[int, int]], float]:
    """Demand-paging baseline: every request's missing pages are fetched
    independently (no cross-request coalescing), in arrival order, until the
    bandwidth budget runs out. Shared prefixes are re-transferred per
    request — the waste the planner's fan-out coalescing avoids."""
    t0 = time.perf_counter()
    budget = cfg.window_bandwidth_bytes
    free_slots = cfg.gpu_capacity_pages - table.gpu_pages()  # post-eviction room
    ops: list[_Op] = []
    for kw in intervals:  # already in request arrival order
        npages = kw["page_end"] - kw["page_start"] + 1
        nbytes = npages * kw["page_bytes"]
        if nbytes > budget or npages > free_slots:
            continue
        budget -= nbytes
        free_slots -= npages
        ops.append(
            _Op(
                pcluster=kw["pcluster"],
                layer=kw["layer"],
                tier_src=kw["tier_src"],
                tier_dst=kw["tier_dst"],
                start_pid=kw["page_start"],
                end_pid=kw["page_end"],
                page_bytes=kw["page_bytes"],
            )
        )
    ms = (time.perf_counter() - t0) * 1000.0
    return ops, [], ms


def _background_evict(table: _PageTable, cfg: TieringConfig, page_bytes: int) -> int:
    """Node-agent-style eviction shared by both policies: keep GPU under the
    watermark by dropping coldest pages first. Returns bytes evicted."""
    target = int(cfg.gpu_capacity_pages * cfg.evict_watermark)
    over = table.gpu_pages() - target
    if over <= 0:
        return 0
    residents = [(k, table.hits.get(k, 0)) for k, t in table.tiers.items() if t == 2]
    residents.sort(key=lambda kv: kv[1])
    evicted = 0
    for (s, layer, lp), _ in residents[:over]:
        if table.tiers.pop((s, layer, lp), None) is not None:
            evicted += page_bytes
    return evicted


def run_tiering_experiment(
    trace: WorkloadTrace,
    tier_cfg: TieringConfig,
    policy: str = "planner",
) -> TieringResult:
    """Run the full simulation. policy is 'planner' or 'naive'."""
    assert policy in ("planner", "naive")
    tcfg = trace.config
    table = _PageTable()
    result = TieringResult(policy=policy)
    page_bytes = tcfg.page_bytes

    for w, requests in enumerate(trace.windows()):
        now_ms = w * tcfg.window_ms
        # ---- 1. score requests against current GPU residency ----
        hit_pages = prefix_hit_pages = 0
        hit_total = prefix_total = 0
        for req in requests:
            for layer in range(tcfg.n_layers):
                for lp in req.page_ids():
                    key = (req.session_id, layer, lp)
                    tier = table.get(key)
                    hit_total += 1
                    if tier == 2:
                        hit_pages += 1
                        table.hits[key] = table.hits.get(key, 0) + 1
                    if lp < req.n_prefix_pages:
                        prefix_total += 1
                        if tier == 2:
                            prefix_hit_pages += 1

        # ---- 2. background eviction (node agent), shared by both policies ----
        bg_evicted = _background_evict(table, tier_cfg, page_bytes)

        # ---- 3. build planner input: one interval set per (request, layer),
        #        grouped by current tier. The planner itself coalesces a
        #        session's shared prefix across its requests (fan-out).
        intervals: list[dict] = []
        for req in requests:
            by_tier: dict[int, list[int]] = {}
            for layer in range(tcfg.n_layers):
                by_tier.clear()
                for lp in req.page_ids():
                    tier = table.get((req.session_id, layer, lp))
                    if tier != 2:
                        by_tier.setdefault(tier, []).append(gid(req.session_id, lp))
                for tier_src, flats in by_tier.items():
                    for start, end in _coalesce(sorted(flats)):
                        pcluster = (
                            req.session_id
                            if tier_cfg.pcluster_mode == "session"
                            else req.req_id
                        )
                        intervals.append(
                            {
                                "req_id": f"r{req.req_id}",
                                "node": "node-0",
                                "model_id": tcfg.model_id,
                                "model_version": tcfg.model_version,
                                "prefix_id": f"session-{req.session_id}",
                                "layer": layer,
                                "page_start": start,
                                "page_end": end,
                                "tier_src": tier_src,
                                "tier_dst": 2,
                                "deadline_ms": req.deadline_ms,
                                "page_bytes": page_bytes,
                                "tenant": "default",
                                "est_fill_ms": 1.0,
                                "pcluster": pcluster,
                            }
                        )

        heat = [
            HeatEntry(
                layer=layer,
                page_id=gid(s, lp),
                decay_hits=table.hits.get((s, layer, lp), 0),
                size_bytes=page_bytes,
            )
            for (s, layer, lp), tier in table.tiers.items()
            if tier == 2
        ]
        gpu_free = max(tier_cfg.gpu_capacity_pages - table.gpu_pages(), 0)
        tier_caps = [
            TierCapacity(tier=0, bandwidth_caps=1 << 60, free_bytes=1 << 60),
            TierCapacity(tier=1, bandwidth_caps=1 << 60, free_bytes=1 << 60),
            TierCapacity(
                tier=2,
                bandwidth_caps=tier_cfg.window_bandwidth_bytes,
                free_bytes=gpu_free * page_bytes,
            ),
        ]

        # ---- 4. plan ----
        if not intervals:
            ops, evictions, ms = [], [], 0.0
        elif policy == "planner":
            ops, evictions, ms = _planner_policy(
                intervals, heat, tier_caps, now_ms, tier_cfg
            )
        else:
            ops, evictions, ms = _naive_policy(intervals, table, tier_cfg)

        # ---- 5. apply ops + evictions to the page table ----
        planned_bytes = 0
        for op in ops:
            s = op.pcluster if tier_cfg.pcluster_mode == "session" else None
            for flat in range(op.start_pid, op.end_pid + 1):
                sess, lp = ungid(flat)
                if s is not None:
                    assert sess == s, "pcluster/session mismatch"
                table.tiers[(sess, op.layer, lp)] = op.tier_dst
            planned_bytes += op.nbytes
        evicted_bytes = bg_evicted
        for layer, flat in evictions:
            s, lp = ungid(flat)
            if table.tiers.pop((s, layer, lp), None) is not None:
                evicted_bytes += page_bytes

        # Post-plan: did the plan serve this window's interactive requests?
        # A request counts as served when >=95% of its pages are GPU-resident
        # (graded hit rate alone rewards broad-but-shallow coverage, which
        # still stalls the request on the missing pages).
        inter_hit = inter_total = 0
        inter_served = inter_n = 0
        for req in requests:
            if not req.is_interactive:
                continue
            inter_n += 1
            pages = [
                (req.session_id, layer, lp)
                for layer in range(tcfg.n_layers)
                for lp in req.page_ids()
            ]
            resident = sum(1 for key in pages if table.get(key) == 2)
            inter_hit += resident
            inter_total += len(pages)
            if resident / len(pages) >= 0.95:
                inter_served += 1

        result.windows.append(
            WindowMetrics(
                window=w,
                n_requests=len(requests),
                hit_rate=hit_pages / hit_total if hit_total else 0.0,
                prefix_hit_rate=prefix_hit_pages / prefix_total
                if prefix_total
                else 0.0,
                interactive_hit_rate=inter_hit / inter_total if inter_total else 0.0,
                interactive_served_rate=inter_served / inter_n if inter_n else 0.0,
                planned_bytes=planned_bytes,
                evicted_bytes=evicted_bytes,
                n_ops=len(ops),
                gpu_pages=table.gpu_pages(),
                planner_ms=ms,
            )
        )
        result.total_planned_bytes += planned_bytes
        result.total_evicted_bytes += evicted_bytes
        if w == tcfg.n_windows - 1:
            result.last_ops = ops

    return result


def _summarize_one(res: TieringResult) -> dict:
    n = max(len(res.windows), 1)
    return {
        "mean_hit_rate": round(res.mean_hit_rate, 4),
        "mean_prefix_hit_rate": round(res.mean_prefix_hit_rate, 4),
        "mean_interactive_hit_rate": round(res.mean_interactive_hit_rate, 4),
        "mean_interactive_served_rate": round(res.mean_interactive_served_rate, 4),
        "total_planned_gb": round(res.total_planned_bytes / 1e9, 3),
        "total_evicted_gb": round(res.total_evicted_bytes / 1e9, 3),
        "mean_ops": round(sum(w.n_ops for w in res.windows) / n, 1),
        "mean_planner_ms": round(sum(w.planner_ms for w in res.windows) / n, 2),
    }


def summarize_comparison(planner: TieringResult, naive: TieringResult) -> dict:
    return {"planner": _summarize_one(planner), "naive": _summarize_one(naive)}
