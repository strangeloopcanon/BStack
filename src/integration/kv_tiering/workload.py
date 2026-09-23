"""Synthetic LLM-serving workload trace for KV-cache tiering experiments.

The trace models the access pattern that makes KV caching worthwhile: many
requests share a long system prompt (a *prefix*), so the KV pages for that
prefix are hot and worth keeping in the fast tier, while each request also
brings unique context pages that are cold.

Every request needs, per layer, the pages ``0 .. n_pages-1`` where pages
``0 .. n_prefix_pages-1`` are the session's shared prefix (identical page ids
across all requests of the session) and the rest are request-unique.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TraceConfig:
    n_sessions: int = 40
    n_requests: int = 600
    n_layers: int = 8
    tokens_per_page: int = 16
    # System-prefix length in pages per session (shared across its requests).
    prefix_pages_lo: int = 8
    prefix_pages_hi: int = 32
    # Unique context pages per request, Zipf-distributed (a few long, many short).
    zipf_a: float = 1.6
    unique_pages_max: int = 48
    n_windows: int = 12
    window_ms: int = 50
    page_bytes: int = 256 * 1024  # KV bytes per page per layer
    model_id: str = "demo-llm"
    model_version: str = "v3"
    seed: int = 7


@dataclass
class Request:
    req_id: int
    session_id: int
    window: int
    n_prefix_pages: int
    n_unique_pages: int
    deadline_ms: int
    is_interactive: bool = False

    @property
    def n_pages(self) -> int:
        return self.n_prefix_pages + self.n_unique_pages

    def page_ids(self) -> range:
        """All page ids this request needs (prefix pages first, shared ids)."""
        return range(self.n_pages)


@dataclass
class WorkloadTrace:
    config: TraceConfig
    requests: list[Request] = field(default_factory=list)
    # session_id -> number of shared prefix pages
    session_prefix_pages: dict[int, int] = field(default_factory=dict)

    def windows(self) -> list[list[Request]]:
        out: list[list[Request]] = [[] for _ in range(self.config.n_windows)]
        for req in self.requests:
            out[req.window].append(req)
        return out


def generate_trace(config: TraceConfig) -> WorkloadTrace:
    """Generate a deterministic, seeded workload trace."""
    rng = np.random.default_rng(config.seed)
    trace = WorkloadTrace(config=config)

    prefix_pages = rng.integers(
        config.prefix_pages_lo, config.prefix_pages_hi + 1, size=config.n_sessions
    )
    for s, p in enumerate(prefix_pages):
        trace.session_prefix_pages[s] = int(p)

    # Spread requests over windows; shuffle sessions so prefixes interleave.
    sessions = rng.integers(0, config.n_sessions, size=config.n_requests)
    windows = np.array_split(rng.permutation(config.n_requests), config.n_windows)
    req_window = np.empty(config.n_requests, dtype=int)
    for w, idxs in enumerate(windows):
        req_window[idxs] = w

    order = rng.permutation(config.n_requests)
    # A quarter of the sessions are interactive-only (their own users); the
    # rest are batch. Interactive sessions share no prefix pages with batch
    # sessions, so serving batch demand does not accidentally help them.
    n_inter_sessions = max(1, config.n_sessions // 4)
    inter_sessions = set(
        rng.choice(config.n_sessions, size=n_inter_sessions, replace=False).tolist()
    )
    for i in order:
        s = int(sessions[i])
        w = int(req_window[i])
        # Zipf(1.6): heavy tail of long contexts, like real serving traces.
        unique = int(rng.zipf(config.zipf_a))
        unique = min(max(unique - 1, 1), config.unique_pages_max)
        is_inter = s in inter_sessions
        # Interactive requests have tight deadlines (must be served now);
        # batch requests are lax. The planner prioritizes by deadline.
        deadline = (
            w * config.window_ms + config.window_ms // 2
            if is_inter
            else (w + 3) * config.window_ms
        )
        trace.requests.append(
            Request(
                req_id=i,
                session_id=s,
                window=w,
                n_prefix_pages=int(prefix_pages[s]),
                n_unique_pages=unique,
                deadline_ms=int(deadline),
                is_interactive=is_inter,
            )
        )
    # Within a window, interactive requests arrive LAST (adversarial to FIFO:
    # a demand-paging baseline serves whoever arrived first).
    trace.requests.sort(key=lambda r: (r.window, r.is_interactive, r.req_id))
    return trace


def trace_stats(trace: WorkloadTrace) -> dict:
    cfg = trace.config
    n_pages_total = sum(r.n_pages for r in trace.requests) * cfg.n_layers
    # Distinct (session, page) pairs = the true working set.
    distinct: set[tuple[int, int]] = set()
    for r in trace.requests:
        for p in r.page_ids():
            distinct.add((r.session_id, p))
    reuse = n_pages_total / max(len(distinct) * cfg.n_layers, 1)
    return {
        "n_requests": len(trace.requests),
        "n_sessions": cfg.n_sessions,
        "n_layers": cfg.n_layers,
        "page_requests_total": n_pages_total,
        "distinct_session_pages": len(distinct),
        "mean_reuse_per_page": round(reuse, 2),
    }
