from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional


class TransferKind(str, Enum):
    """Enumerates high-level data movement operations."""

    H2D = "H2D"
    D2H = "D2H"
    P2P = "P2P"
    STORAGE2H = "STORAGE2H"

    @classmethod
    def from_string(cls, value: str | None) -> "TransferKind":
        if not value:
            return cls.STORAGE2H
        try:
            return cls[value]
        except KeyError:
            return cls[value.upper()]


@dataclass
class KvPageRef:
    tensor: str
    page: int
    head: int
    layer: int


@dataclass
class TransferOp:
    kind: TransferKind
    src: str
    dst: str
    length: int
    src_offset: int = 0
    dst_offset: int = 0
    kv_refs: List[KvPageRef] = field(default_factory=list)
    note: str | None = None


@dataclass
class FileChunk:
    path: str
    offset: int
    length: int
    sha256: str


@dataclass
class WeightManifest:
    model_id: str
    version: str
    files: List[FileChunk]


@dataclass
class SwapWindow:
    t_start_ns: int
    t_deadline_ns: int


@dataclass
class CachePlan:
    plan_id: str
    ops: List[TransferOp]
    prefetch: List[KvPageRef] = field(default_factory=list)
    evict: List[KvPageRef] = field(default_factory=list)

    def to_dict(self) -> dict:
        return _to_dict(self)

    def to_json(self, path: Path | str | None = None, *, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            Path(path).write_text(payload)
        return payload


@dataclass
class SwapPlan:
    plan_id: str
    manifest_from: WeightManifest
    manifest_to: WeightManifest
    ops: List[TransferOp]
    window: SwapWindow

    def to_dict(self) -> dict:
        return _to_dict(self)

    def to_json(self, path: Path | str | None = None, *, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            Path(path).write_text(payload)
        return payload


# -----------------------------------------------------------------------------
# Helpers


def _to_dict(obj) -> dict:
    return _convert(obj)


def _convert(value):
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, list):
        return [_convert(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_convert(v) for v in value)
    if isinstance(value, dict):
        return {k: _convert(v) for k, v in value.items()}
    if dataclass_isinstance(value):
        return {f.name: _convert(getattr(value, f.name)) for f in fields(value)}
    return value


def dataclass_isinstance(obj) -> bool:
    return hasattr(obj, "__dataclass_fields__")


# -----------------------------------------------------------------------------
# Factory functions


def kv_ref(**kwargs) -> KvPageRef:
    return KvPageRef(**kwargs)


def transfer_op(kind: str | TransferKind, **kwargs) -> TransferOp:
    if isinstance(kind, str):
        kind = TransferKind.from_string(kind)
    return TransferOp(kind=kind, **kwargs)


def cache_plan(plan_id: str, ops: Iterable[TransferOp], *, prefetch: Iterable[KvPageRef] | None = None, evict: Iterable[KvPageRef] | None = None) -> CachePlan:
    return CachePlan(plan_id=plan_id, ops=list(ops), prefetch=list(prefetch or []), evict=list(evict or []))


def swap_plan(
    plan_id: str,
    manifest_from: WeightManifest,
    manifest_to: WeightManifest,
    ops: Iterable[TransferOp],
    *,
    window: SwapWindow,
) -> SwapPlan:
    return SwapPlan(plan_id=plan_id, manifest_from=manifest_from, manifest_to=manifest_to, ops=list(ops), window=window)


def weight_manifest(model_id: str, version: str, files: Iterable[FileChunk]) -> WeightManifest:
    return WeightManifest(model_id=model_id, version=version, files=list(files))


def file_chunk(path: str, offset: int, length: int, sha256: str) -> FileChunk:
    return FileChunk(path=path, offset=offset, length=length, sha256=sha256)


def swap_window(start_ns: int, deadline_ns: int) -> SwapWindow:
    return SwapWindow(t_start_ns=start_ns, t_deadline_ns=deadline_ns)


def load_cache_plan(path: Path | str) -> CachePlan:
    data = json.loads(Path(path).read_text())
    return _cache_plan_from_dict(data)


def load_swap_plan(path: Path | str) -> SwapPlan:
    data = json.loads(Path(path).read_text())
    return _swap_plan_from_dict(data)


# -----------------------------------------------------------------------------
# Deserialisation


def _kv_ref_from_dict(payload: dict) -> KvPageRef:
    return KvPageRef(
        tensor=str(payload.get("tensor", "")),
        page=int(payload.get("page", 0)),
        head=int(payload.get("head", 0)),
        layer=int(payload.get("layer", 0)),
    )


def _transfer_op_from_dict(payload: dict) -> TransferOp:
    return TransferOp(
        kind=TransferKind.from_string(str(payload.get("kind", "STORAGE2H"))),
        src=str(payload.get("src", "")),
        dst=str(payload.get("dst", "")),
        length=int(payload.get("length", 0)),
        src_offset=int(payload.get("src_offset", 0)),
        dst_offset=int(payload.get("dst_offset", 0)),
        kv_refs=[_kv_ref_from_dict(ref) for ref in payload.get("kv_refs", [])],
        note=payload.get("note"),
    )


def _manifest_from_dict(payload: dict) -> WeightManifest:
    return WeightManifest(
        model_id=str(payload.get("model_id", "")),
        version=str(payload.get("version", "")),
        files=[
            FileChunk(
                path=str(f.get("path", "")),
                offset=int(f.get("offset", 0)),
                length=int(f.get("length", 0)),
                sha256=str(f.get("sha256", "")),
            )
            for f in payload.get("files", [])
        ],
    )


def _swap_plan_from_dict(payload: dict) -> SwapPlan:
    return SwapPlan(
        plan_id=str(payload.get("plan_id", "")),
        manifest_from=_manifest_from_dict(payload.get("manifest_from", payload.get("from", {}))),
        manifest_to=_manifest_from_dict(payload.get("manifest_to", payload.get("to", {}))),
        ops=[_transfer_op_from_dict(op) for op in payload.get("ops", [])],
        window=SwapWindow(
            t_start_ns=int(payload.get("window", {}).get("t_start_ns", payload.get("window", {}).get("start_ns", 0))),
            t_deadline_ns=int(payload.get("window", {}).get("t_deadline_ns", payload.get("window", {}).get("deadline_ns", 0))),
        ),
    )


def _cache_plan_from_dict(payload: dict) -> CachePlan:
    return CachePlan(
        plan_id=str(payload.get("plan_id", "")),
        ops=[_transfer_op_from_dict(op) for op in payload.get("ops", [])],
        prefetch=[_kv_ref_from_dict(ref) for ref in payload.get("prefetch", [])],
        evict=[_kv_ref_from_dict(ref) for ref in payload.get("evict", [])],
    )


__all__ = [
    "TransferKind",
    "KvPageRef",
    "TransferOp",
    "CachePlan",
    "SwapPlan",
    "WeightManifest",
    "SwapWindow",
    "FileChunk",
    "kv_ref",
    "transfer_op",
    "cache_plan",
    "swap_plan",
    "weight_manifest",
    "file_chunk",
    "swap_window",
    "load_cache_plan",
    "load_swap_plan",
]
