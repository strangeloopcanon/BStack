from __future__ import annotations

import pandas as pd

from datajax.api import djit
from datajax.frame.frame import Frame


@djit
def _feature_pipeline(df: Frame) -> Frame:
    # Multiply then aggregate to demonstrate trace recording
    doubled = (df["tokens"] * 2).rename("tokens2")
    aggregated = doubled.groupby(df["user"]).sum()
    return aggregated


def sample_feature_plan() -> dict:
    """Run a small DataJAX pipeline and return a human-readable summary."""

    data = pd.DataFrame(
        {
            "user": ["a", "b", "a", "c"],
            "tokens": [12, 4, 8, 7],
        }
    )
    result = _feature_pipeline(data)
    exec_record = _feature_pipeline.last_execution
    assert exec_record is not None
    plan = exec_record.plan
    summary = {
        "backend": exec_record.backend,
        "mode": exec_record.backend_mode,
        "stages": plan.describe(),
        "output_preview": result.to_pandas().reset_index().to_dict(orient="records"),
    }
    return summary
