#!/usr/bin/env python3
from __future__ import annotations

import configparser
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


def get_submodules() -> list[tuple[str, Path]]:
    """Parse .gitmodules and return a list of (name, path) tuples."""
    gitmodules_path = ROOT / ".gitmodules"
    if not gitmodules_path.exists():
        return []

    config = configparser.ConfigParser()
    config.read(gitmodules_path)

    modules = []
    for section in config.sections():
        path_str = config.get(section, "path", fallback=None)
        if path_str:
            path = ROOT / path_str
            name = path.name
            modules.append((name, path))
    return modules


def current_shas() -> dict[str, str]:
    """Get the current HEAD SHA for each submodule."""
    out: dict[str, str] = {}
    modules = get_submodules()
    for name, path in modules:
        if not path.exists():
            continue
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
