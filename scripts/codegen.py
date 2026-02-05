#!/usr/bin/env python3
"""Generate protobuf bindings for bstack_apis."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "src" / "bstack_apis" / "proto" / "plan.proto"
OUT_PY = ROOT / "src" / "bstack_apis" / "python"


def run_protoc() -> None:
    if not PROTO.exists():
        raise SystemExit(f"Missing proto file: {PROTO}")
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO.parent}",
        f"--python_out={OUT_PY}",
        str(PROTO),
    ]
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError as exc:  # pragma: no cover - guard
        raise SystemExit("grpcio-tools is not installed; install via `pip install -e .[dev]`") from exc


if __name__ == "__main__":
    run_protoc()
