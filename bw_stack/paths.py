from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"


def add_third_party_to_path() -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(THIRD_PARTY / "BCache"))
    sys.path.insert(0, str(THIRD_PARTY / "hotweights"))
    sys.path.insert(0, str(THIRD_PARTY / "datajax"))
    sys.path.insert(0, str(THIRD_PARTY / "bw-runtime" / "python"))


def resolve(*parts: str) -> Path:
    return ROOT.joinpath(*parts)
