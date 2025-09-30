#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "stack.lock"


def parse_lock() -> dict[str, str]:
    if not LOCK.exists():
        raise SystemExit("stack.lock missing; run `git submodule update --init --recursive` first")
    entries: dict[str, str] = {}
    for line in LOCK.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sha = [part.strip() for part in line.split("=", 1)]
        entries[name.replace("_", "-")] = sha.strip("\"')")
    return entries


def current_shas() -> dict[str, str]:
    out: dict[str, str] = {}
    modules = [
        ("hotweights", ROOT / "third_party" / "hotweights"),
        ("BCache", ROOT / "third_party" / "BCache"),
        ("datajax", ROOT / "third_party" / "datajax"),
        ("bw-runtime", ROOT / "third_party" / "bw-runtime"),
    ]
    for name, path in modules:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path).decode().strip()
        out[name] = sha
    return out


def main():
    lock = parse_lock()
    current = current_shas()
    mismatches = []
    for name, sha in current.items():
        in_lock = lock.get(name)
        if not in_lock:
            mismatches.append(f"{name}: missing from lock file")
            continue
        if in_lock != sha:
            mismatches.append(f"{name}: lock={in_lock[:8]} current={sha[:8]}")
    for name in lock:
        if name not in current:
            mismatches.append(f"{name}: present in lock but submodule missing")
    if mismatches:
        joined = "\n  ".join(mismatches)
        raise SystemExit(f"Submodule lock mismatch:\n  {joined}\nRun `make sync` after updating stack.lock")
    print("stack.lock matches submodules")


if __name__ == "__main__":
    main()
