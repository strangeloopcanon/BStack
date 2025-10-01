from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from bw_apis import (
    FileChunk,
    SwapPlan,
    SwapWindow,
    TransferKind,
    TransferOp,
    WeightManifest,
    file_chunk,
    swap_plan,
)
from bw_stack.paths import add_third_party_to_path

add_third_party_to_path()

from hotweights.manifest import build_simple_manifest
from hotweights.core.replicate import create_plan


@dataclass
class SwapPlanResult:
    plan: SwapPlan
    prev_manifest: WeightManifest
    next_manifest: WeightManifest
    buckets: list[dict]


def build_swap_plan(
    prev_checkpoint: Path | str,
    next_checkpoint: Path | str,
    *,
    model_id: str = "demo",
    prev_version: str = "prev",
    next_version: str = "next",
    bucket_mb: int = 32,
    deadline_ns: Optional[int] = None,
) -> SwapPlanResult:
    """Produce a SwapPlan by diffing two checkpoint directories."""

    os.environ.setdefault("HOTWEIGHTS_FORCE_PANDAS", "1")

    prev_manifest_raw = build_simple_manifest(model_id=model_id, version=prev_version, checkpoint_dir=str(prev_checkpoint))
    next_manifest_raw = build_simple_manifest(model_id=model_id, version=next_version, checkpoint_dir=str(next_checkpoint))

    prev_manifest = _to_weight_manifest(prev_manifest_raw)
    next_manifest = _to_weight_manifest(next_manifest_raw)

    bucket_plan = create_plan(prev_manifest_raw, next_manifest_raw, bucket_mb=bucket_mb)
    buckets = list(bucket_plan.get("buckets", []))
    plan_id = f"swap-{next_manifest.version}"
    start_ns = time.time_ns()
    deadline_ns = deadline_ns if deadline_ns is not None else start_ns + 5_000_000_000  # +5s

    ops: list[TransferOp] = []
    for bucket in buckets:
        bucket_id = int(bucket.get("bucket_id", 0))
        for item in bucket.get("items", []):
            uri = str(item.get("uri", ""))
            length = int(item.get("nbytes", 0))
            offset = int(item.get("offset", 0))
            tensor = item.get("tensor", "tensor")
            shard = int(item.get("shard_rank", 0))
            dst = f"device://bucket/{bucket_id}"
            note = f"tensor={tensor} shard={shard} bucket={bucket_id} offset={offset}"
            ops.append(
                TransferOp(
                    kind=TransferKind.STORAGE2H,
                    src=uri,
                    dst=dst,
                    length=length,
                    src_offset=offset,
                    dst_offset=offset,
                    kv_refs=[],
                    note=note,
                )
            )

    swap = swap_plan(
        plan_id,
        prev_manifest,
        next_manifest,
        ops,
        window=SwapWindow(t_start_ns=start_ns, t_deadline_ns=deadline_ns),
    )

    return SwapPlanResult(plan=swap, prev_manifest=prev_manifest, next_manifest=next_manifest, buckets=buckets)


def _to_weight_manifest(manifest: dict) -> WeightManifest:
    model_id = manifest.get("model_id", "model")
    version = manifest.get("version", "0")
    files: list[FileChunk] = []
    for tensor in manifest.get("tensors", []):
        for shard in tensor.get("shards", []):
            uri = str(shard.get("uri", ""))
            path = uri
            if uri.startswith("file://"):
                path = uri[len("file://") :]
            files.append(
                file_chunk(
                    path=path,
                    offset=0,
                    length=int(shard.get("bytes", 0)),
                    sha256=str(shard.get("hash", "")),
                )
            )
    return WeightManifest(model_id=model_id, version=version, files=files)


def bucket_summary(buckets: Iterable[dict]) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for bucket in buckets:
        out.append(
            {
                "bucket_id": int(bucket.get("bucket_id", 0)),
                "items": len(bucket.get("items", [])),
                "bytes": int(bucket.get("size", 0)),
            }
        )
    return out
