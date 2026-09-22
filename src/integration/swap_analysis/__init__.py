"""Weight-swap x KV-cache invalidation analysis (hotweights + BCache IR)."""

from integration.swap_analysis.swap_kv import (
    SwapKvResult,
    analyze_swap_kv,
    layer_of_tensor,
    make_checkpoints,
    report_dict,
)

__all__ = [
    "SwapKvResult",
    "analyze_swap_kv",
    "layer_of_tensor",
    "make_checkpoints",
    "report_dict",
]
