from __future__ import annotations

import json
from pathlib import Path

from bw_apis import (
    CachePlan,
    FileChunk,
    KvPageRef,
    SwapPlan,
    SwapWindow,
    TransferKind,
    TransferOp,
    WeightManifest,
    cache_plan,
    load_cache_plan,
    load_swap_plan,
    swap_plan,
)


def test_cache_plan_round_trip(tmp_path: Path) -> None:
    refs = [KvPageRef(tensor="qkv", page=1, head=2, layer=3)]
    ops = [
        TransferOp(
            kind=TransferKind.H2D,
            src="hbm://node0/page1",
            dst="gpu://0/0",
            length=4096,
            kv_refs=refs,
        )
    ]
    plan = cache_plan("window-1", ops, prefetch=refs, evict=[])
    out = tmp_path / "cache_plan.json"
    plan.to_json(out)
    restored = load_cache_plan(out)
    assert isinstance(restored, CachePlan)
    assert restored.plan_id == "window-1"
    assert restored.ops[0].kind == TransferKind.H2D
    assert restored.prefetch[0].tensor == "qkv"


def test_swap_plan_round_trip(tmp_path: Path) -> None:
    manifest_from = WeightManifest(
        model_id="m",
        version="v0",
        files=[FileChunk(path="a", offset=0, length=16, sha256="00")],
    )
    manifest_to = WeightManifest(
        model_id="m",
        version="v1",
        files=[FileChunk(path="a", offset=0, length=16, sha256="11")],
    )
    plan = swap_plan(
        "swap-1",
        manifest_from,
        manifest_to,
        [
            TransferOp(
                kind=TransferKind.STORAGE2H,
                src="file://a",
                dst="hbm://0",
                length=16,
            )
        ],
        window=SwapWindow(t_start_ns=0, t_deadline_ns=1_000_000),
    )
    out = tmp_path / "swap_plan.json"
    plan.to_json(out)
    restored = load_swap_plan(out)
    assert isinstance(restored, SwapPlan)
    assert restored.plan_id == "swap-1"
    assert restored.manifest_to.version == "v1"
    assert restored.ops[0].length == 16
    assert restored.window.t_deadline_ns == 1_000_000
