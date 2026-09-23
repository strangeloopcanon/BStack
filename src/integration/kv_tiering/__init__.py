"""KV-cache hierarchical tiering demo: realistic workload -> BCache planner -> scored placement."""

from integration.kv_tiering.tiering import (
    TieringConfig,
    TieringResult,
    run_tiering_experiment,
)
from integration.kv_tiering.workload import (
    TraceConfig,
    WorkloadTrace,
    generate_trace,
    trace_stats,
)

__all__ = [
    "TieringConfig",
    "TieringResult",
    "TraceConfig",
    "WorkloadTrace",
    "generate_trace",
    "run_tiering_experiment",
    "trace_stats",
]
