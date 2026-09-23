"""Weight-swap x KV-cache invalidation analysis.

A weight update invalidates the KV cache — but only for the layers whose
tensors actually changed. This module quantifies that: it builds two
synthetic checkpoints (v1 = v0 with a subset of layers re-trained), runs
hotweights' swap planner to find exactly which tensors changed, maps those
tensors back to layers, and reports which KV pages can be *reused* versus
must be invalidated.

The invalidation set is exported as a ``CachePlan`` (evict list) in the
shared ``bstack_apis`` IR, so a serving system could consume it directly.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bstack.paths import add_third_party_to_path

add_third_party_to_path()

os.environ.setdefault("HOTWEIGHTS_FORCE_PANDAS", "1")

from hotweights.core.replicate import create_plan
from hotweights.manifest import build_simple_manifest

from bstack_apis import CachePlan, KvPageRef, cache_plan

_LAYER_RE = re.compile(r"layer(\d+)")


def layer_of_tensor(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def make_checkpoints(
    root: Path,
    n_layers: int = 8,
    changed_layers: tuple[int, ...] = (2, 5),
    seed: int = 11,
    dim: int = 64,
) -> tuple[Path, Path]:
    """Write v0/v1 checkpoint dirs; v1 re-randomizes `changed_layers`."""
    rng = np.random.default_rng(seed)
    v0, v1 = root / "ckpt_v0", root / "ckpt_v1"
    for d in (v0, v1):
        d.mkdir(parents=True, exist_ok=True)
    for layer in range(n_layers):
        for suffix in ("qkv", "mlp"):
            w = rng.standard_normal((dim, dim)).astype(np.float32)
            np.save(v0 / f"layer{layer}.{suffix}.npy", w)
            if layer in changed_layers:
                w2 = rng.standard_normal((dim, dim)).astype(np.float32)
            else:
                w2 = w  # bit-identical -> no delta
            np.save(v1 / f"layer{layer}.{suffix}.npy", w2)
    return v0, v1


@dataclass
class SwapKvResult:
    plan: CachePlan  # evict list = KV pages invalidated by the swap
    changed_layers: list[int] = field(default_factory=list)
    changed_tensors: list[str] = field(default_factory=list)
    swap_bytes: int = 0
    kv_page_bytes: int = 0
    kv_pages_per_layer: int = 0
    n_layers: int = 0
    kv_invalidated_pages: int = 0
    kv_reusable_pages: int = 0

    @property
    def kv_invalidated_bytes(self) -> int:
        return self.kv_invalidated_pages * self.kv_page_bytes

    @property
    def kv_reusable_bytes(self) -> int:
        return self.kv_reusable_pages * self.kv_page_bytes

    @property
    def kv_total_bytes(self) -> int:
        return self.n_layers * self.kv_pages_per_layer * self.kv_page_bytes


def analyze_swap_kv(
    prev_ckpt: Path,
    next_ckpt: Path,
    *,
    model_id: str = "demo-llm",
    n_layers: int = 8,
    kv_pages_per_layer: int = 64,
    kv_page_bytes: int = 256 * 1024,
    bucket_mb: int = 32,
) -> SwapKvResult:
    prev = build_simple_manifest(
        model_id=model_id, version="v0", checkpoint_dir=str(prev_ckpt)
    )
    nxt = build_simple_manifest(
        model_id=model_id, version="v1", checkpoint_dir=str(next_ckpt)
    )
    bucket_plan = create_plan(prev, nxt, bucket_mb=bucket_mb)

    changed_tensors: list[str] = []
    swap_bytes = 0
    for bucket in bucket_plan.get("buckets", []):
        for item in bucket.get("items", []):
            changed_tensors.append(str(item.get("tensor", "")))
            swap_bytes += int(item.get("nbytes", 0))

    changed_layers = sorted(
        {l for t in changed_tensors if (l := layer_of_tensor(t)) is not None}
    )

    evict_refs: list[KvPageRef] = []
    invalidated = reusable = 0
    for layer in range(n_layers):
        for page in range(kv_pages_per_layer):
            ref = KvPageRef(tensor="kv", page=page, head=0, layer=layer)
            if layer in changed_layers:
                evict_refs.append(ref)
                invalidated += 1
            else:
                reusable += 1

    plan = cache_plan("swap-kv-impact-v0-v1", [], evict=evict_refs)
    return SwapKvResult(
        plan=plan,
        changed_layers=changed_layers,
        changed_tensors=sorted(set(changed_tensors)),
        swap_bytes=swap_bytes,
        kv_page_bytes=kv_page_bytes,
        kv_pages_per_layer=kv_pages_per_layer,
        n_layers=n_layers,
        kv_invalidated_pages=invalidated,
        kv_reusable_pages=reusable,
    )


def report_dict(result: SwapKvResult) -> dict:
    total = result.kv_total_bytes or 1
    return {
        "model_id": "demo-llm",
        "n_layers": result.n_layers,
        "changed_layers": result.changed_layers,
        "changed_tensors": result.changed_tensors,
        "swap_bytes": result.swap_bytes,
        "kv_total_mb": round(total / 1e6, 2),
        "kv_invalidated_mb": round(result.kv_invalidated_bytes / 1e6, 2),
        "kv_reusable_mb": round(result.kv_reusable_bytes / 1e6, 2),
        "reuse_fraction": round(result.kv_reusable_bytes / total, 4),
        "full_flush_mb": round(total / 1e6, 2),
        "saved_vs_full_flush_mb": round((total - result.kv_invalidated_bytes) / 1e6, 2),
    }
