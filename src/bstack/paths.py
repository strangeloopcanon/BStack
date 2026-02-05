from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve(*parts: str) -> Path:
    """Resolve a path relative to the repository root."""
    return ROOT.joinpath(*parts)


def add_third_party_to_path() -> None:
    """Ensure submodule Python packages are importable in local dev."""
    for path in (
        resolve("third_party", "BCache"),
        resolve("third_party", "hotweights"),
        resolve("third_party", "datajax"),
        resolve("third_party", "bw-runtime", "python"),
    ):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


__all__ = ["ROOT", "resolve", "add_third_party_to_path"]
