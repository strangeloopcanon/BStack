from __future__ import annotations

import sys
from pathlib import Path

# tests live outside tests/; make src importable on our own
SRC = Path(__file__).resolve().parents[2]
src_str = str(SRC)
if src_str not in sys.path:
    sys.path.insert(0, src_str)
